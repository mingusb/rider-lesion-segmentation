#pragma once

#include <openssl/evp.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace dp_streaming {

// File identity metadata only; this helper performs no model computation.
inline constexpr std::size_t file_buffer_bytes = 64 * 1024;

inline std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open for SHA256: " + path.string());

    using Context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
    Context context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("cannot initialize file SHA256");

    std::array<char, file_buffer_bytes> buffer{};
    for (;;) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        // Never publish a digest of an unreadable or partially failed stream.
        if (input.bad() || (input.fail() && !input.eof()))
            throw std::runtime_error("cannot complete file SHA256: " + path.string());
        if (count > 0 && EVP_DigestUpdate(context.get(), buffer.data(),
                                        static_cast<std::size_t>(count)) != 1)
            throw std::runtime_error("cannot update file SHA256");
        if (input.eof()) break;
    }

    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned length = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1 || length != 32)
        throw std::runtime_error("cannot finalize file SHA256");

    constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (std::size_t i = 0; i < length; ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 15];
    }
    return result;
}

} // namespace dp_streaming
