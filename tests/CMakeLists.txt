include_directories(${CMAKE_SOURCE_DIR}/src)

# add_executable(test_tokenizer test_tokenizer.cpp)
# target_link_libraries(test_tokenizer PRIVATE bert ggml)

set(TEST_MODEL_NAME "bge-large-zh-v1.5")

function(bert_build_executable source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_executable(${TEST_TARGET} ${source})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE bert ggml)
endfunction()

function(bert_test_executable name source)
    get_filename_component(TEST_TARGET ${source} NAME_WE)
    add_test(NAME "Generate_HF_tokens" COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_hf_tokenizer.py ${TEST_MODEL_NAME})
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${TEST_TARGET}> ${ARGN})
    set_property(TEST ${name} PROPERTY LABELS "main")
endfunction()


bert_build_executable(test_tokenizer.cpp)
bert_test_executable (test_tokenizer test_tokenizer.cpp -m ${CMAKE_CURRENT_SOURCE_DIR}/../models/${TEST_MODEL_NAME}/bge-large-zh-v1.5-q4_1.gguf)
