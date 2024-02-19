#ifdef NDEBUG
#undef NDEBUG
#endif

#include "bert.h"
#include "ggml.h"

#include <cassert>
#include <unistd.h>
#include <map>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_GREEN   "\x1b[32m"


std::vector<std::string> txt2list(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::string> all_lines;

    if (!file.is_open()) {
        printf("can not open file: %s\n", filename.c_str());
        return all_lines;
    }

    std::string line;
    while (std::getline(file, line)) {
        all_lines.push_back(line);
    }

    file.close();
    return all_lines;
}

std::vector<std::vector<int>> read_expected_tokenids(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<int>> all_numbers;

    if (!file.is_open()) {
        printf("can not open file: %s\n", filename.c_str());
        return all_numbers;
    }


    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> line_numbers;
        std::istringstream iss(line);
        std::string number_str;

        while (std::getline(iss, number_str, ',')) {
            line_numbers.push_back(std::stoi(number_str));
        }

        all_numbers.push_back(line_numbers);
    }

    file.close();
    return all_numbers;
}

void tokenizer_test(bert_ctx * ctx, const std::string& input, const bert_tokens& expected) {
    int N = bert_n_max_tokens(ctx);
    bert_tokens result = bert_tokenize(ctx, input, N);
    int n_tokens;

    if (result != expected) {
        printf("tokenizer test failed: '%.*s'\n", 16000, input.data());

        printf("[");
        for (auto& tok : result) {
            printf("%d, ", tok);
        }
        printf("]\n");

        for (size_t i = 0; i < result.size(); i++) {
            bert_token a = expected[std::min(i, expected.size()-1)];
            bert_token b = result[i];
            const char *color_start = (a == b) ? ANSI_COLOR_GREEN : ANSI_COLOR_RED;
            const char *color_end = ANSI_COLOR_RESET;

            printf("%s%d -> %s : %d -> %s%s\n", color_start, a, bert_vocab_id_to_token(ctx, a), b, bert_vocab_id_to_token(ctx, b), color_end);
        }
    } else {
        printf("Success '%.*s...'\n", 16, input.data());
    }
    assert(result == expected);
  }


struct bert_params
{
    int32_t n_threads = 6;
    const char* model = "models/all-MiniLM-L6-v2/ggml-model-q4_0.bin";
    const char* prompt = "test prompt";
    int32_t batch_size = 32;
    bool use_cpu = false;
};

void bert_print_usage(char **argv, const bert_params &params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model);
    fprintf(stderr, "                        batch size to use when executing model\n");
    fprintf(stderr, "  -c, --cpu             use CPU backend (default: use CUDA if available)\n");
    fprintf(stderr, "\n");
}

bool bert_params_parse(int argc, char **argv, bert_params &params) {
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-c" || arg == "--cpu") {
            params.use_cpu = true;
        } else if (arg == "-h" || arg == "--help") {
            bert_print_usage(argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bert_print_usage(argv, params);
            exit(0);
        }
    }

    return true;
}

int main(int argc, char ** argv) {

    bert_params params;
    params.model = "models/all-MiniLM-L6-v2/ggml-model-q4_0.bin";

    if (bert_params_parse(argc, argv, params) == false) {
        return 1;
    }


    bert_ctx * bctx;

    // load the model
    {
        if ((bctx = bert_load_from_file(params.model, params.use_cpu)) == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
            return 1;
        }
    }
    std::string dir = params.model;
    std::size_t i = dir.rfind("/models/");
    if (i != std::string::npos) {
        dir.resize(i);
    } else {
        dir = ".";
    }

    auto expected = read_expected_tokenids(dir + "/tests/hf_tokenized_ids.txt");
    auto prompts = txt2list(dir + "/tests/test_prompts.txt");

    if (expected.size() == 0 || prompts.size() == 0) {
        printf("failed to read test data\n");
        return 1;
    }

    if (expected.size() != prompts.size()) {
        printf("test data size mismatch\n");
        return 1;
    }

    // tokenizer tests:
    for (size_t i = 0; i < prompts.size(); i++) {
        tokenizer_test(bctx, prompts[i], expected[i]);
    }

    // tokenizer_test(bctx, "1231 2431431", {101, 13138, 2487, 22884, 16932, 21486, 102});
    // tokenizer_test(bctx, "Québec", {101, 5447, 102});
    // tokenizer_test(bctx, "syömme \t  täällä tänään", {101, 25353, 5358, 4168, 11937, 25425, 9092, 14634, 102});
    // tokenizer_test(bctx, "I'm going to the store to buy 3 apples and a banana! You're welcome to come along if you'd like. The time is 2:30 p.m. and it's partly cloudy outside. I'll be back soon, so don't go anywhere.", {101, 1045, 1005, 1049, 2183, 2000, 1996, 3573, 2000, 4965, 1017, 18108, 1998, 1037, 15212, 999, 2017, 1005, 2128, 6160, 2000, 2272, 2247, 2065, 2017, 1005, 1040, 2066, 1012, 1996, 2051, 2003, 1016, 1024, 2382, 1052, 1012, 1049, 1012, 1998, 2009, 1005, 1055, 6576, 24706, 2648, 1012, 1045, 1005, 2222, 2022, 2067, 2574, 1010, 2061, 2123, 1005, 1056, 2175, 5973, 1012, 102});
    // tokenizer_test(bctx, "\"5 2 + 3 * 4 -\"; int stack[1000], top = -1; int calculate(int a, int b, char operator) { return operator == '+' ? a + b : operator == '-' ? a - b : operator == '*' ? a * b : a / b; } void push(int x) { stack[++top] = x; } int pop() { return stack[top--]; } int evaluatePostfix(char* expression) { for (int i = 0; expression[i]; i++) { if (isdigit(expression[i])) push(expression[i] - '0'); else { int a = pop(), b = pop(); push(calculate(b, a, expression[i])); } } return pop(); } int result = evaluatePostfix(input);", {101, 1000, 1019, 1016, 1009, 1017, 1008, 1018, 1011, 1000, 1025, 20014, 9991, 1031, 6694, 1033, 1010, 2327, 1027, 1011, 1015, 1025, 20014, 18422, 1006, 20014, 1037, 1010, 20014, 1038, 1010, 25869, 6872, 1007, 1063, 2709, 6872, 1027, 1027, 1005, 1009, 1005, 1029, 1037, 1009, 1038, 1024, 6872, 1027, 1027, 1005, 1011, 1005, 1029, 1037, 1011, 1038, 1024, 6872, 1027, 1027, 1005, 1008, 1005, 1029, 1037, 1008, 1038, 1024, 1037, 1013, 1038, 1025, 1065, 11675, 5245, 1006, 20014, 1060, 1007, 1063, 9991, 1031, 1009, 1009, 2327, 1033, 1027, 1060, 1025, 1065, 20014, 3769, 1006, 1007, 1063, 2709, 9991, 1031, 2327, 1011, 1011, 1033, 1025, 1065, 20014, 16157, 19894, 8873, 2595, 1006, 25869, 1008, 3670, 1007, 1063, 2005, 1006, 20014, 1045, 1027, 1014, 1025, 3670, 1031, 1045, 1033, 1025, 1045, 1009, 1009, 1007, 1063, 2065, 1006, 2003, 4305, 23806, 1006, 3670, 1031, 1045, 1033, 1007, 1007, 5245, 1006, 3670, 1031, 1045, 1033, 1011, 1005, 1014, 1005, 1007, 1025, 2842, 1063, 20014, 1037, 1027, 3769, 1006, 1007, 1010, 1038, 1027, 3769, 1006, 1007, 1025, 5245, 1006, 18422, 1006, 1038, 1010, 1037, 1010, 3670, 1031, 1045, 1033, 1007, 1007, 1025, 1065, 1065, 2709, 3769, 1006, 1007, 1025, 1065, 20014, 2765, 1027, 16157, 19894, 8873, 2595, 1006, 7953, 1007, 1025, 102});

    // tokenizer_test(bctx, "Hello world!", {101, 7592, 2088,  999,  102});
    // tokenizer_test(bctx, "你好，世界！", {101, 100, 100, 1989, 1745, 100, 1986, 102});
    // tokenizer_test(bctx, "こんにちは、世界！", {101,  1655, 30217, 30194, 30188, 30198, 1635, 1745, 100, 1986, 102});
}
