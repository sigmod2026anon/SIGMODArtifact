#include "poisoning/gen_real.h"
#include "common/binary_io.h"
#include <iostream>
#include <fstream>

namespace poisoning {

template<typename T>
std::vector<T> RealDataSampler::sample_from_dataset(const std::vector<T>& original_data, 
                                                    size_t n, 
                                                    std::uint64_t seed) {
    if (original_data.size() < n) {
        throw std::runtime_error("Sample size n is larger than dataset size");
    }
    
    rng_.seed(seed);
    
    std::uniform_int_distribution<size_t> dist(0, original_data.size() - n);
    size_t start = dist(rng_);
    
    std::vector<T> result(original_data.begin() + start, 
                         original_data.begin() + start + n);
    
    return result;
}

void sample_and_save_real_datasets(const RealSamplingConfig& config) {
    sample_and_save_real_datasets(config, "../../../data/");
}

void sample_and_save_real_datasets(const RealSamplingConfig& config, const std::string& output_dir) {
    RealDataSampler sampler;
    
    for (const auto& dataset_info : config.datasets) {
        // Build path of dataset file
        std::string original_dataset_file = "../../../../data/" + dataset_info.name + "_" + dataset_info.dtype_str;
        
        // Check if file exists (using ifstream)
        std::ifstream test_file(original_dataset_file);
        if (!test_file.good()) {
            std::cerr << "Warning: Dataset file not found: " << original_dataset_file << std::endl;
            std::cerr << "Skipping dataset: " << dataset_info.name << "_" << dataset_info.dtype_str << std::endl;
            continue;
        }
        test_file.close();
        
        // Read file once
        if (dataset_info.dtype_str == "uint32") {
            std::cerr << "Loading dataset: " << original_dataset_file << std::endl;
            const std::vector<std::uint32_t> original_data = common::read_from_binary<std::uint32_t>(original_dataset_file);
            
            for (size_t n : config.ns) {
                for (std::uint64_t seed : config.seeds) {
                    try {
                        std::string output_file = output_dir + dataset_info.name + "_n" + std::to_string(n) + 
                                                "_seed" + std::to_string(seed) + "_" + dataset_info.dtype_str;
                        std::ifstream test_output(output_file);
                        if (test_output.good()) {
                            test_output.close();
                            std::cerr << "[Skipped] Output file already exists: " << output_file << std::endl;
                            continue;
                        }
                        std::cerr << "Outputting to " << output_file << std::endl;

                        // Sample from loaded data
                        std::vector<std::uint32_t> sampled_data = sampler.sample_from_dataset(original_data, n, seed);
                        
                        common::write_to_binary(sampled_data, output_file);
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Error processing " << dataset_info.name << "_n" << n 
                                 << "_seed" << seed << "_" << dataset_info.dtype_str << ": " << e.what() << std::endl;
                        continue;
                    }
                }
            }
        } else { // uint64
            std::cerr << "Loading dataset: " << original_dataset_file << std::endl;
            const std::vector<std::uint64_t> original_data = common::read_from_binary<std::uint64_t>(original_dataset_file);
            
            for (size_t n : config.ns) {
                for (std::uint64_t seed : config.seeds) {
                    try {
                        std::string output_file = output_dir + dataset_info.name + "_n" + std::to_string(n) + 
                                                "_seed" + std::to_string(seed) + "_" + dataset_info.dtype_str;
                        std::ifstream test_output(output_file);
                        if (test_output.good()) {
                            test_output.close();
                            std::cerr << "[Skipped] Output file already exists: " << output_file << std::endl;
                            continue;
                        }
                        std::cerr << "Outputting to " << output_file << std::endl;

                        // Sample from loaded data
                        std::vector<std::uint64_t> sampled_data = sampler.sample_from_dataset(original_data, n, seed);
                        
                        common::write_to_binary(sampled_data, output_file);
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Error processing " << dataset_info.name << "_n" << n 
                                 << "_seed" << seed << "_" << dataset_info.dtype_str << ": " << e.what() << std::endl;
                        continue;
                    }
                }
            }
        }
        
        // Message when each dataset is completed (moved outside n and seed loops)
        // for (size_t n : config.ns) {
        //     std::cout << dataset_info.name << "_n" << n << "_seed[0-9]+_" << dataset_info.dtype_str << std::endl;
        // }
    }
}

// Explicit template instantiation
template std::vector<std::uint32_t> RealDataSampler::sample_from_dataset<std::uint32_t>(
    const std::vector<std::uint32_t>&, size_t, std::uint64_t);
template std::vector<std::uint64_t> RealDataSampler::sample_from_dataset<std::uint64_t>(
    const std::vector<std::uint64_t>&, size_t, std::uint64_t);

} // namespace poisoning 