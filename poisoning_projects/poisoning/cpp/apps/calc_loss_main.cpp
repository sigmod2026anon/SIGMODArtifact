#include "common/binary_io.h"
#include "poisoning/calc_loss.h"
#include "common/types.h"
#include "common/dataset_info.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <sstream>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <json_output_file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " data/books_200M_n100_seed0_uint64 results/loss/result.json" << std::endl;
        std::cerr << "Output: JSON format with keys: dataset_name, n, R, seed, data_type, lambda, loss" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    std::string json_output_file = argv[2];
    
    try {
        // Infer data type
        auto data_type = common::infer_data_type(filename);
        
        double loss;
        
        if (data_type == common::DataType::UINT32) {
            auto data = common::read_uint32_from_binary(filename);
            std::vector<std::uint64_t> xs(data.begin(), data.end());
            loss = poisoning::calc_loss(xs);
        } else { // UINT64
            auto data = common::read_uint64_from_binary(filename);
            loss = poisoning::calc_loss(data);
        }
        
        // Extract detailed information from filename
        std::string basename = std::filesystem::path(filename).filename().string();
        common::DatasetInfo info = common::parse_filename(basename);
        
        // Build data in JSON format
        std::ostringstream json_entry;
        json_entry << "{\n"
                  << "  \"dataset_name\": \"" << info.dataset_name << "\",\n"
                  << "  \"n\": " << info.n << ",\n"
                  << "  \"R\": " << info.R << ",\n"
                  << "  \"seed\": " << info.seed << ",\n"
                  << "  \"data_type\": \"" << info.data_type << "\",\n"
                  << "  \"lambda\": " << info.lambda << ",\n"
                  << "  \"loss\": " << std::fixed << std::setprecision(15) << loss << "\n"
                  << "}";
        
        // Write to JSON file (as individual files)
        std::ofstream json_file(json_output_file);
        if (json_file.is_open()) {
            json_file << json_entry.str();
            json_file.close();
        } else {
            std::cerr << "Error: Cannot open JSON output file: " << json_output_file << std::endl;
            return 1;
        }
        
        // Output for progress display
        // std::cout << "Added entry to " << json_output_file << ": " 
        //          << info.dataset_name << " (lambda=" << info.lambda << ")" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}