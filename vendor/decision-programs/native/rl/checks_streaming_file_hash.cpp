#include "streaming_file_hash.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

std::string one_shot(std::string_view bytes) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned length = 0;
    if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &length,
                   EVP_sha256(), nullptr) != 1 || length != 32)
        throw std::runtime_error("reference SHA256 failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned i = 0; i < length; ++i) {
        result.push_back(hex[digest[i] >> 4]);
        result.push_back(hex[digest[i] & 15]);
    }
    return result;
}

void write_file(const fs::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) throw std::runtime_error("fixture write failed");
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("need NEW_FIXTURE_DIRECTORY PINNED_XGBOOST_LIBRARY");
        const fs::path directory(argv[1]);
        if (!fs::create_directory(directory)) throw std::runtime_error("fixture directory must be fresh");
        std::uint64_t assertions = 0, rejects = 0;
        auto need = [&](bool condition, const char* message) {
            if (!condition) throw std::runtime_error(message);
            ++assertions;
        };
        auto expect_failure = [&](const fs::path& path) {
            bool failed = false;
            try { (void)dp_streaming::sha256_file(path); }
            catch (const std::exception&) { failed = true; }
            need(failed, "failed input published a digest");
            ++rejects;
        };

        write_file(directory / "empty", "");
        need(dp_streaming::sha256_file(directory / "empty") ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
             "empty published SHA differs");
        write_file(directory / "abc", "abc");
        need(dp_streaming::sha256_file(directory / "abc") ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
             "abc published SHA differs");

        constexpr std::size_t lengths[] = {1, 55, 56, 63, 64, 65, 255, 256,
                                           65535, 65536, 65537, 131072, 131089};
        for (const auto length : lengths) {
            std::string bytes(length, '\0');
            for (std::size_t i = 0; i < length; ++i)
                bytes[i] = static_cast<char>((i * 37 + (i >> 8)) & 255);
            const auto path = directory / ("binary-" + std::to_string(length));
            write_file(path, bytes);
            const auto original = dp_streaming::sha256_file(path);
            need(original == one_shot(bytes), "binary or block-boundary digest differs");
            bytes.back() = static_cast<char>(static_cast<unsigned char>(bytes.back()) ^ 128);
            write_file(path, bytes);
            need(dp_streaming::sha256_file(path) == one_shot(bytes), "mutated digest differs");
            need(dp_streaming::sha256_file(path) != original, "mutation went undetected");
        }
        expect_failure(directory / "missing");
        expect_failure(directory);

        const auto library_hash = dp_streaming::sha256_file(argv[2]);
        need(library_hash ==
             "462aa6331ecb178df8d10f16612c865ae70f571c248138e50c8a9eb4bc007dd4",
             "full pinned native library digest differs");
        std::cout << "{\"passed\":true,\"assertions\":" << assertions
                  << ",\"rejects\":" << rejects
                  << ",\"file_buffer_bytes\":" << dp_streaming::file_buffer_bytes
                  << ",\"pinned_library_bytes\":" << fs::file_size(argv[2])
                  << ",\"pinned_library_sha256\":\"" << library_hash
                  << "\",\"CUDA_executed\":false,\"scope\":\"file identity metadata\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
