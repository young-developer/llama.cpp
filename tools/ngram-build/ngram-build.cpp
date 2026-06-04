#include "common.h"
#include "log.h"
#include "llama.h"
#include "ngram-mod.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void print_usage(const char * argv0) {
    printf("usage: %s [options]\n\n", argv0);
    printf("Build an ngram-mod cache file from source code.\n\n");
    printf("Options:\n");
    printf("  -h, --help                    print this help and exit\n");
    printf("  --validate FILE               validate an existing .ngram cache file and print info\n");
    printf("  --invalidate FILE             remove a stale .ngram cache file\n");
    printf("  -m MODEL, --model MODEL       path to GGUF model (tokenizer only)\n");
    printf("  -d DIR, --dir DIR             input directory\n");
    printf("  --ext EXT                     file extension to include (repeatable, default: .cpp)\n");
    printf("  -o FILE, --output FILE        output .ngram file\n");
    printf("  --n-match N                   n-gram size (default: 24)\n");
    printf("  --n-max N                     max speculative tokens (default: 64)\n");
    printf("  --n-min N                     min speculative tokens (default: 48)\n");
    printf("  --recursive                   recurse into subdirectories\n");
    printf("  --scan-count N                scan files N times to accumulate frequencies (default: 1)\n");
    printf("  --keep-top PCT                keep top PCT%% of entries by frequency (default: 30)\n");
    printf("  --no-prune                    disable pruning; keep all entries\n");
    printf("  --with-path                   also tokenize file paths into ngram-mod\n");
    printf("\n");
}

static std::string read_file(const std::string & path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "error: cannot open '%s'\n", path.c_str());
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string dir_path;
    std::string output_path;
    std::string validate_path;
    std::string invalidate_path;
    std::vector<std::string> extensions;
    int32_t n_match = 24;
    int32_t n_max   = 64;
    int32_t n_min   = 48;
    bool recursive = false;
    int32_t scan_count = 1;
    double keep_top_pct = 30.0;
    bool no_prune = false;
    bool with_path = false;

    bool model_set = false, dir_set = false, output_set = false, ext_set = false, validate_set = false, invalidate_set = false;

    for (int i = 1; i < argc; i++) {
        std::string arg{argv[i]};

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else       if (arg == "--validate") {
            if (validate_set) { fprintf(stderr, "error: --validate specified multiple times\n"); return 1; }
            validate_path = argv[++i]; validate_set = true;
        } else if (arg == "--invalidate") {
            if (invalidate_set) { fprintf(stderr, "error: --invalidate specified multiple times\n"); return 1; }
            invalidate_path = argv[++i]; invalidate_set = true;
        } else if (arg == "-m" || arg == "--model") {
            if (model_set) { fprintf(stderr, "error: -m specified multiple times\n"); return 1; }
            model_path = argv[++i]; model_set = true;
        } else if (arg == "-d" || arg == "--dir") {
            if (dir_set) { fprintf(stderr, "error: -d specified multiple times\n"); return 1; }
            dir_path = argv[++i]; dir_set = true;
        } else if (arg == "-o" || arg == "--output") {
            if (output_set) { fprintf(stderr, "error: -o specified multiple times\n"); return 1; }
            output_path = argv[++i]; output_set = true;
        } else if (arg == "--ext") {
            extensions.push_back(argv[++i]); ext_set = true;
        } else if (arg == "--n-match") {
            n_match = std::stoi(argv[++i]);
        } else if (arg == "--n-max") {
            n_max = std::stoi(argv[++i]);
        } else if (arg == "--n-min") {
            n_min = std::stoi(argv[++i]);
        } else if (arg == "--recursive") {
            recursive = true;
        } else if (arg == "--scan-count") {
            scan_count = std::stoi(argv[++i]);
        } else if (arg == "--keep-top") {
            keep_top_pct = std::stod(argv[++i]);
        } else if (arg == "--no-prune") {
            no_prune = true;
        } else if (arg == "--with-path") {
            with_path = true;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", arg.c_str());
            return 1;
        }
    }

    if (!model_set) { fprintf(stderr, "error: must specify -m (model)\n"); return 1; }
    if (!dir_set)    { fprintf(stderr, "error: must specify -d (directory)\n"); return 1; }
    if (!output_set) { fprintf(stderr, "error: must specify -o (output)\n"); return 1; }
    if (!ext_set)    { extensions.push_back(".cpp"); }

    // Validate mode: check an existing cache file
    if (validate_set) {
        if (!fs::exists(validate_path)) {
            fprintf(stderr, "error: file not found: '%s'\n", validate_path.c_str());
            return 1;
        }
        try {
            std::string loaded_tokenizer_name;
            int32_t loaded_n_match = 0, loaded_n_max = 0, loaded_n_min = 0;
            auto loaded = common_ngram_mod_load(validate_path, nullptr, &loaded_tokenizer_name,
                                                  &loaded_n_match, &loaded_n_max, &loaded_n_min);
            printf("file:       '%s'\n", validate_path.c_str());
            printf("version:    %d\n", 1);
            printf("n_match:    %d\n", loaded_n_match);
            printf("n_max:      %d\n", loaded_n_max);
            printf("n_min:      %d\n", loaded_n_min);
            printf("tokenizer:  '%s'\n", loaded_tokenizer_name.c_str());
            printf("ngrams:     %zu / %zu (%.2f%%)\n",
                    loaded.get_used(), loaded.size(),
                    (double)loaded.get_used() / (double)loaded.size() * 100);
            int32_t min_freq = 0, max_freq = 0;
            int64_t sum_freq = 0;
            loaded.get_freq_stats(&min_freq, &max_freq, &sum_freq);
            printf("freq:       min=%d max=%d sum=%lld\n", min_freq, max_freq, (long long)sum_freq);
            // File size
            std::ifstream sz(validate_path, std::ios::binary | std::ios::ate);
            if (sz) {
                printf("size:       %.2f MB\n", (double)sz.tellg() / 1024 / 1024);
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    // Invalidate mode: remove a stale cache file
    if (invalidate_set) {
        if (!fs::exists(invalidate_path)) {
            fprintf(stderr, "error: file not found: '%s'\n", invalidate_path.c_str());
            return 1;
        }
        std::error_code ec;
        fs::remove(invalidate_path, ec);
        if (ec) {
            fprintf(stderr, "error: failed to remove '%s': %s\n", invalidate_path.c_str(), ec.message().c_str());
            return 1;
        }
        printf("removed: '%s'\n", invalidate_path.c_str());
        return 0;
    }

    if (!fs::exists(model_path)) {
        fprintf(stderr, "error: model file not found: '%s'\n", model_path.c_str());
        return 1;
    }
    if (!fs::is_directory(dir_path)) {
        fprintf(stderr, "error: not a directory: '%s'\n", dir_path.c_str());
        return 1;
    }

    // Load model for tokenizer
    common_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;

    auto model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    char tk_buf[128];
    std::string tokenizer_name;
    if (llama_model_meta_val_str(model, "tokenizer.ggml.model", tk_buf, sizeof(tk_buf)) > 0) {
        tokenizer_name = tk_buf;
    }

    // Collect files
    std::vector<std::string> files;
    if (recursive) {
        for (auto & entry : fs::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::error_code ec;
                auto ext = entry.path().extension().string();
                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    files.push_back(entry.path().string());
                }
            }
        }
    } else {
        for (auto & entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::error_code ec;
                auto ext = entry.path().extension().string();
                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    files.push_back(entry.path().string());
                }
            }
        }
    }

    // Sort for deterministic output
    std::sort(files.begin(), files.end());

    LOG_INF("processing %d file(s) from '%s'\n", (int)files.size(), dir_path.c_str());

    // Create ngram-mod
    const size_t mod_size = 4 * 1024 * 1024;
    common_ngram_mod mod(static_cast<uint16_t>(n_match), mod_size);

    // Process each file, scan_count times
    for (int32_t scan = 1; scan <= scan_count; scan++) {
        if (scan_count > 1) {
            LOG_INF("  scan %d/%d\n", scan, scan_count);
        }

        for (const auto & filepath : files) {
            std::string text = read_file(filepath);
            if (text.empty()) continue;

            // Tokenize file content
            std::vector<llama_token> tokens = common_tokenize(vocab, text, true, false);

            if (with_path) {
                // Tokenize file path and prepend to tokens
                std::string path_str = filepath;
                std::vector<llama_token> path_tokens = common_tokenize(vocab, path_str, false, false);
                tokens.insert(tokens.begin(), path_tokens.begin(), path_tokens.end());
            }

            if (tokens.size() < (size_t)n_match) continue;

            // Feed to ngram-mod
            for (size_t i = 0; i < tokens.size() - n_match; i++) {
                mod.add(tokens.data() + i);
            }

            if (scan_count == 1) {
                LOG_INF("  %s: %d tokens\n", fs::path(filepath).filename().string().c_str(), (int)tokens.size());
            }
        }
    }

      // Print stats
    {
        int32_t min_freq = 0, max_freq = 0;
        int64_t sum_freq = 0;
        mod.get_freq_stats(&min_freq, &max_freq, &sum_freq);
        double avg_freq = mod.get_used() > 0 ? (double)sum_freq / (double)mod.get_used() : 0.0;
        LOG_INF("unique ngrams: %zu / %zu (%.2f%%), freq: min=%d max=%d avg=%.2f\n",
                mod.get_used(), mod.size(),
                (double)mod.get_used() / (double)mod.size() * 100,
                min_freq, max_freq, avg_freq);
    }

    // Prune to keep only top PCT% by frequency
    if (!no_prune && keep_top_pct < 100.0) {
        const size_t before = mod.get_used();
        const size_t removed = mod.prune_keep_top(keep_top_pct / 100.0);
        LOG_INF("pruned: %zu removed, %zu kept (%.1f%% of %zu)\n",
                removed, mod.get_used(), keep_top_pct, before);
    }

    // Normalize frequencies to 1 before saving (server will re-accumulate from actual usage)
    mod.normalize_freq();

 // Save
    try {
        common_ngram_mod_save(mod, output_path, "", tokenizer_name,
                                n_match, n_max, n_min);

        // Validate the saved file immediately
        if (!common_ngram_mod_validate(output_path, "", tokenizer_name,
                                        n_match, n_max, n_min)) {
            fprintf(stderr, "error: saved file failed validation - output may be corrupted\n");
            llama_model_free(model);
            return 1;
        }

        double occ = (double)mod.get_used() / (double)mod.size() * 100;
        LOG_INF("saved to '%s': %zu/%zu used (%.2f%%)\n",
                output_path.c_str(), mod.get_used(), mod.size(), occ);

        // Print file size
        std::ifstream check(output_path, std::ios::binary);
        if (check) {
            check.seekg(0, std::ios::end);
            size_t fsize = check.tellg();
            LOG_INF("file size: %.2f MB\n", (double)fsize / 1024 / 1024);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: failed to save: %s\n", e.what());
        llama_model_free(model);
        return 1;
    }

    llama_model_free(model);
    return 0;
}
