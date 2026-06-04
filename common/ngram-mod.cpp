#include "ngram-mod.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstring>
#include <utility>

// Get the size of the file in bytes, or -1 on error
static int64_t get_file_size(const std::string & filename) {
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs) return -1;
    return ifs.tellg();
}

//
// common_ngram_mod
//

// Magic header for ngram-mod binary files: "NGRM" (0x4E47524D)
static constexpr uint32_t NGRAM_MOD_MAGIC = 0x4E47524D;
static constexpr uint32_t NGRAM_MOD_VERSION = 1;

struct ngram_mod_header {
    uint32_t magic;
    uint32_t version;
    uint64_t n;
    uint64_t used;
    uint64_t entries_size;
    int32_t  n_match;
    int32_t  n_max;
    int32_t  n_min;
    uint64_t tokenizer_name_len;
};

static bool read_ngram_mod_header(std::ifstream & ifs, ngram_mod_header & hdr) {
    if (!ifs.read(reinterpret_cast<char *>(&hdr.magic), sizeof(hdr.magic))) return false;
    if (hdr.magic != NGRAM_MOD_MAGIC) return false;
    if (!ifs.read(reinterpret_cast<char *>(&hdr.version), sizeof(hdr.version))) return false;
    if (!ifs.read(reinterpret_cast<char *>(&hdr.n), sizeof(hdr.n))) return false;
    if (!ifs.read(reinterpret_cast<char *>(&hdr.used), sizeof(hdr.used))) return false;
    if (!ifs.read(reinterpret_cast<char *>(&hdr.entries_size), sizeof(hdr.entries_size))) return false;
    if (!ifs.read(reinterpret_cast<char *>(&hdr.n_match), sizeof(hdr.n_match))) return false;
    if (!ifs.read(reinterpret_cast<char *>(&hdr.n_max), sizeof(hdr.n_max))) return false;
    if (!ifs.read(reinterpret_cast<char *>(&hdr.n_min), sizeof(hdr.n_min))) return false;
    if (!ifs.read(reinterpret_cast<char *>(&hdr.tokenizer_name_len), sizeof(hdr.tokenizer_name_len))) return false;
    return true;
}

common_ngram_mod::common_ngram_mod(uint16_t n, size_t cap) : n(n), used(0) {
    // Round up to power of 2 for fast modulo via bitmask
    size_t sz = 1;
    while (sz < cap) sz <<= 1;
    mask = sz - 1;
    entries.resize(sz);
    freq.resize(sz);
    accept_count.resize(sz);
    reject_count.resize(sz);

    reset();
}

void common_ngram_mod_save(const common_ngram_mod & ngram_mod, const std::string & filename,
                               const std::string & model_path, const std::string & tokenizer_name,
                               int32_t n_match, int32_t n_max, int32_t n_min) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("failed to open file for writing: " + filename);
    }

    // Write magic header
    ofs.write(reinterpret_cast<const char *>(&NGRAM_MOD_MAGIC), sizeof(NGRAM_MOD_MAGIC));
    ofs.write(reinterpret_cast<const char *>(&NGRAM_MOD_VERSION), sizeof(NGRAM_MOD_VERSION));

    // Write n (ngram size)
    uint64_t n_val = static_cast<uint64_t>(ngram_mod.get_n());
    ofs.write(reinterpret_cast<const char *>(&n_val), sizeof(n_val));

    // Write used
    uint64_t used_val = ngram_mod.get_used();
    ofs.write(reinterpret_cast<const char *>(&used_val), sizeof(used_val));

    // Write entries size
    uint64_t entries_size = ngram_mod.size();
    ofs.write(reinterpret_cast<const char *>(&entries_size), sizeof(entries_size));

    // Write nmod config
    ofs.write(reinterpret_cast<const char *>(&n_match), sizeof(n_match));
    ofs.write(reinterpret_cast<const char *>(&n_max), sizeof(n_max));
    ofs.write(reinterpret_cast<const char *>(&n_min), sizeof(n_min));

    // Write tokenizer name (8-byte length + content)
    uint64_t tk_len = tokenizer_name.size();
    ofs.write(reinterpret_cast<const char *>(&tk_len), sizeof(tk_len));
    if (!tokenizer_name.empty()) {
        ofs.write(tokenizer_name.c_str(), tk_len);
    }

   // Write only used entries: [4B index][4B token][4B freq][2B accept_count][2B reject_count]
    const auto & entries = ngram_mod.entries;
    const auto & freq = ngram_mod.freq;
    const auto & accept_count = ngram_mod.accept_count;
    const auto & reject_count = ngram_mod.reject_count;
    for (size_t i = 0; i < entries_size; i++) {
        if (entries[i] != common_ngram_mod::EMPTY) {
            uint32_t idx = static_cast<uint32_t>(i);
            ofs.write(reinterpret_cast<const char *>(&idx), sizeof(idx));
            ofs.write(reinterpret_cast<const char *>(&entries[i]), sizeof(entries[i]));
            ofs.write(reinterpret_cast<const char *>(&freq[i]), sizeof(freq[i]));
            ofs.write(reinterpret_cast<const char *>(&accept_count[i]), sizeof(accept_count[i]));
            ofs.write(reinterpret_cast<const char *>(&reject_count[i]), sizeof(reject_count[i]));
        }
    }

    ofs.close();
    if (!ofs) {
        throw std::runtime_error("failed to write ngram-mod to file: " + filename);
    }
}

bool common_ngram_mod_validate(const std::string & filename,
                                     const std::string & model_path, const std::string & tokenizer_name,
                                     int32_t n_match, int32_t n_max, int32_t n_min) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        fprintf(stderr, "warning: ngram-mod cannot open file: %s\n", filename.c_str());
        return false;
    }

    ngram_mod_header hdr;
    if (!read_ngram_mod_header(ifs, hdr)) {
        fprintf(stderr, "warning: ngram-mod invalid header in file: %s\n", filename.c_str());
        return false;
    }

    if (hdr.version != NGRAM_MOD_VERSION) {
        fprintf(stderr, "warning: ngram-mod unsupported version! File: %d, expected: %d\n",
                hdr.version, NGRAM_MOD_VERSION);
        return false;
    }

    if (hdr.n_match != n_match || hdr.n_max != n_max || hdr.n_min != n_min) {
        fprintf(stderr, "warning: ngram-mod config mismatch! File: n_match=%d n_max=%d n_min=%d, "
                "current: n_match=%d n_max=%d n_min=%d\n",
                hdr.n_match, hdr.n_max, hdr.n_min, n_match, n_max, n_min);
        return false;
    }

    // Verify file is large enough to contain the metadata
    uint64_t expected_size = sizeof(ngram_mod_header) + hdr.tokenizer_name_len;
    int64_t file_size = get_file_size(filename);
    if (file_size < 0 || static_cast<uint64_t>(file_size) < expected_size) {
        fprintf(stderr, "warning: ngram-mod file too small for metadata: %s (expected >= %llu, got %lld)\n",
                filename.c_str(), (unsigned long long)expected_size, file_size);
        return false;
    }

    // Validate tokenizer name
    if (hdr.tokenizer_name_len > 0) {
        std::string file_tokenizer(static_cast<size_t>(hdr.tokenizer_name_len), '\0');
        ifs.read(&file_tokenizer[0], static_cast<std::streamsize>(hdr.tokenizer_name_len));
        if (!ifs) {
            fprintf(stderr, "warning: ngram-mod failed to read tokenizer name from file: %s\n", filename.c_str());
            return false;
        }
        if (!tokenizer_name.empty() && file_tokenizer != tokenizer_name) {
            fprintf(stderr, "warning: ngram-mod tokenizer mismatch! File: '%s', current: '%s'\n",
                    file_tokenizer.c_str(), tokenizer_name.c_str());
            return false;
        }
    } else if (!tokenizer_name.empty()) {
        // File has no tokenizer but current config does
        fprintf(stderr, "warning: ngram-mod tokenizer mismatch! File: <none>, current: '%s'\n",
                tokenizer_name.c_str());
        return false;
    }

    return true;
}

common_ngram_mod common_ngram_mod_load(const std::string & filename,
                                          std::string * out_model_path, std::string * out_tokenizer_name,
                                          int32_t * out_n_match, int32_t * out_n_max, int32_t * out_n_min) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("failed to open file for reading: " + filename);
    }

    ngram_mod_header hdr;
    if (!read_ngram_mod_header(ifs, hdr)) {
        throw std::runtime_error("invalid ngram-mod file: failed to read header");
    }

   if (out_n_match) *out_n_match = hdr.n_match;
    if (out_n_max) *out_n_max = hdr.n_max;
    if (out_n_min) *out_n_min = hdr.n_min;

    if (hdr.tokenizer_name_len > 0) {
        std::string tk(hdr.tokenizer_name_len, '\0');
        ifs.read(&tk[0], hdr.tokenizer_name_len);
        if (out_tokenizer_name) *out_tokenizer_name = std::move(tk);
    }

    common_ngram_mod result(static_cast<uint16_t>(hdr.n), hdr.entries_size);

    for (uint64_t i = 0; i < hdr.used; i++) {
        uint32_t idx = 0;
        common_ngram_mod::entry_t token = common_ngram_mod::EMPTY;
        int32_t freq = 0;
        uint16_t acc_count = 0;
        uint16_t rej_count = 0;
        ifs.read(reinterpret_cast<char *>(&idx), sizeof(idx));
        ifs.read(reinterpret_cast<char *>(&token), sizeof(token));
        ifs.read(reinterpret_cast<char *>(&freq), sizeof(freq));
        ifs.read(reinterpret_cast<char *>(&acc_count), sizeof(acc_count));
        ifs.read(reinterpret_cast<char *>(&rej_count), sizeof(rej_count));
        if (idx < hdr.entries_size) {
            result.entries[idx] = token;
            result.freq[idx] = freq;
            result.accept_count[idx] = acc_count;
            result.reject_count[idx] = rej_count;
        }
    }

    result.used = hdr.used;

    ifs.close();
    if (!ifs) {
        throw std::runtime_error("failed to read ngram-mod from file: " + filename);
    }

    return result;
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    return res & mask;
}

void common_ngram_mod::add(const entry_t * tokens) {
    size_t i = idx(tokens);

    if (entries[i] != EMPTY) {
        entries[i] = tokens[n];
        freq[i] = 1;
        accept_count[i] = 0;
        reject_count[i] = 0;
        return;
    }

    used++;
    freq[i] = 1;
    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    size_t i = idx(tokens);

    if (entries[i] != EMPTY) {
        return entries[i];
    }

    return EMPTY;
}

void common_ngram_mod::inc(const entry_t * tokens) {
    size_t i = idx(tokens);

    if (entries[i] != EMPTY && freq[i] < INT32_MAX) {
        freq[i]++;
        if (accept_count[i] < UINT16_MAX) {
            accept_count[i]++;
        }
    }
}

void common_ngram_mod::dec(const entry_t * tokens) {
    size_t i = idx(tokens);

    if (entries[i] != EMPTY && freq[i] > INT32_MIN) {
        freq[i]--;
        if (reject_count[i] < UINT16_MAX) {
            reject_count[i]++;
        }
    }
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    std::fill(freq.begin(), freq.end(), 0);
    std::fill(accept_count.begin(), accept_count.end(), 0);
    std::fill(reject_count.begin(), reject_count.end(), 0);
    used = 0;
}

void common_ngram_mod::normalize_freq() {
    for (size_t i = 0; i < freq.size(); i++) {
        if (entries[i] != EMPTY) {
            freq[i] = 1;
            accept_count[i] = 0;
            reject_count[i] = 0;
        }
    }
}

size_t common_ngram_mod::prune_evict_negative(int32_t threshold) {
    size_t removed = 0;

    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i] != EMPTY && freq[i] < threshold) {
            entries[i] = EMPTY;
            freq[i] = 0;
            accept_count[i] = 0;
            reject_count[i] = 0;
            used--;
            removed++;
        }
    }

    return removed;
}

static size_t prune_by_cutoff(common_ngram_mod & mod, size_t to_remove) {
    if (to_remove == 0) return 0;

    std::vector<std::pair<int32_t, size_t>> freq_indices;
    freq_indices.reserve(mod.used);
    for (size_t i = 0; i < mod.entries.size(); ++i) {
        if (mod.entries[i] != common_ngram_mod::EMPTY) {
            freq_indices.emplace_back(mod.freq[i], i);
        }
    }

    std::nth_element(freq_indices.begin(), freq_indices.begin() + static_cast<ptrdiff_t>(to_remove),
                     freq_indices.end());

    const int32_t cutoff_freq = freq_indices[static_cast<ptrdiff_t>(to_remove)].first;

    size_t removed = 0;
    for (size_t i = 0; i < mod.entries.size() && removed < to_remove; ++i) {
        if (mod.entries[i] != common_ngram_mod::EMPTY && mod.freq[i] < cutoff_freq) {
            mod.entries[i] = common_ngram_mod::EMPTY;
            mod.freq[i] = 0;
            mod.accept_count[i] = 0;
            mod.reject_count[i] = 0;
            mod.used--;
            removed++;
        }
    }

    for (size_t i = 0; i < mod.entries.size() && removed < to_remove; ++i) {
        if (mod.entries[i] != common_ngram_mod::EMPTY && mod.freq[i] == cutoff_freq) {
            mod.entries[i] = common_ngram_mod::EMPTY;
            mod.freq[i] = 0;
            mod.accept_count[i] = 0;
            mod.reject_count[i] = 0;
            mod.used--;
            removed++;
        }
    }

    return removed;
}

size_t common_ngram_mod::prune(double target_occupancy) {
    const size_t target_used = static_cast<size_t>(entries.size() * target_occupancy);
    if (used <= target_used) {
        return 0;
    }
    return prune_by_cutoff(*this, used - target_used);
}

size_t common_ngram_mod::prune_keep_top(double keep_pct) {
    if (used == 0) {
        return 0;
    }
    const size_t keep = static_cast<size_t>(std::ceil(used * keep_pct));
    if (keep >= used) {
        return 0;
    }
    return prune_by_cutoff(*this, used - keep);
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

float common_ngram_mod::confidence(size_t i) const {
    if (i >= entries.size() || entries[i] == EMPTY) {
        return -1.0f;
    }
    const uint32_t total = accept_count[i] + reject_count[i];
    if (total == 0) {
        return 1.0f; // no data yet — treat as confident (neutral)
    }
    return (float)accept_count[i] / (float)total;
}

void common_ngram_mod::get_freq_stats(int32_t * out_min, int32_t * out_max, int64_t * out_sum) const {
    int32_t min_f = INT32_MAX;
    int32_t max_f = INT32_MIN;
    int64_t sum_f = 0;
    for (size_t i = 0; i < freq.size(); ++i) {
        if (entries[i] != EMPTY) {
            if (freq[i] < min_f) min_f = freq[i];
            if (freq[i] > max_f) max_f = freq[i];
            sum_f += freq[i];
        }
    }
    if (used == 0) {
        min_f = 0;
        max_f = 0;
    }
    if (out_min) *out_min = min_f;
    if (out_max) *out_max = max_f;
    if (out_sum) *out_sum = sum_f;
}

size_t common_ngram_mod::size() const {
    return mask + 1;
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * (sizeof(entries[0]) + sizeof(freq[0]) +
                              sizeof(accept_count[0]) + sizeof(reject_count[0]));
}

void common_ngram_mod::swap(common_ngram_mod & other) {
    std::swap(n, other.n);
    std::swap(mask, other.mask);
    std::swap(used, other.used);
    entries.swap(other.entries);
    freq.swap(other.freq);
    accept_count.swap(other.accept_count);
    reject_count.swap(other.reject_count);
}
