#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
#include <string>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    friend void common_ngram_mod_save(const common_ngram_mod & ngram_mod, const std::string & filename,
                                       const std::string & model_path, const std::string & tokenizer_name,
                                       int32_t n_match, int32_t n_max, int32_t n_min);
    friend common_ngram_mod common_ngram_mod_load(const std::string & filename,
                                                     std::string * out_model_path, std::string * out_tokenizer_name,
                                                     int32_t * out_n_match, int32_t * out_n_max, int32_t * out_n_min);
    friend size_t prune_by_cutoff(common_ngram_mod & mod, size_t to_remove);

    common_ngram_mod(uint16_t n, size_t cap);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found
    void    inc(const entry_t * tokens);       // increment frequency
    void    dec(const entry_t * tokens);       // decrement frequency

    void reset();
    size_t prune(double target_occupancy);
    size_t prune_keep_top(double keep_pct); // keep top P% of entries by frequency
    size_t prune_evict_negative(int32_t threshold); // evict entries with freq < threshold
    void normalize_freq(); // set all frequencies to 1

    size_t get_n()    const;
    size_t get_used() const;
    void   get_freq_stats(int32_t * out_min, int32_t * out_max, int64_t * out_sum) const;

    // Confidence: accept_count / (accept_count + reject_count), 0.0-1.0
    // Returns -1.0 if entry not found or no data
    float confidence(size_t idx) const;

    size_t size()       const;
    size_t size_bytes() const;

    void swap(common_ngram_mod & other);

private:
    size_t n; // ngram size to hash
    size_t mask; // table size - 1 (power-of-2 for fast modulo)

    size_t used;

    std::vector<entry_t> entries;
    mutable std::vector<int32_t> freq; // access frequency per entry (signed: + for correct, - for wrong)
    std::vector<uint16_t> accept_count; // number of accepted tokens per entry
    std::vector<uint16_t> reject_count; // number of rejected tokens per entry
};

// Save an ngram-mod to a file. model_path, tokenizer_name, and nmod config are stored for validation on load.
void common_ngram_mod_save(const common_ngram_mod & ngram_mod, const std::string & filename,
                              const std::string & model_path, const std::string & tokenizer_name,
                              int32_t n_match, int32_t n_max, int32_t n_min);

// Validate the cache file's metadata against current config. Returns true if all checks pass.
bool common_ngram_mod_validate(const std::string & filename,
                                 const std::string & model_path, const std::string & tokenizer_name,
                                 int32_t n_match, int32_t n_max, int32_t n_min);

// Load an ngram-mod saved with common_ngram_mod_save. Returns metadata via out_ parameters.
common_ngram_mod common_ngram_mod_load(const std::string & filename,
                                         std::string * out_model_path, std::string * out_tokenizer_name,
                                         int32_t * out_n_match, int32_t * out_n_max, int32_t * out_n_min);
