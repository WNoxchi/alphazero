add_executable(alphazero_cpp_smoke test_scaffold_smoke.cpp)
target_compile_features(alphazero_cpp_smoke PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(alphazero_cpp_smoke PRIVATE /W4)
else()
    target_compile_options(alphazero_cpp_smoke PRIVATE -Wall -Wextra)
endif()

add_test(NAME cpp_scaffold_smoke COMMAND alphazero_cpp_smoke)

if(ALPHAZERO_HAS_GTEST)
    include(GoogleTest)

    add_executable(alphazero_cpp_tests
        test_game_interfaces.cpp
        test_chess_bitboard.cpp
        test_chess_movegen.cpp
        test_chess_state.cpp
        test_chess_encoding.cpp
        test_go_rules.cpp
        test_go_encoding.cpp
        test_mcts.cpp
        test_eval_queue.cpp
        test_replay_buffer.cpp
        test_arena.cpp
    )
    target_compile_features(alphazero_cpp_tests PRIVATE cxx_std_20)
    target_link_libraries(alphazero_cpp_tests PRIVATE alphazero_engine GTest::gtest_main)

    gtest_discover_tests(alphazero_cpp_tests DISCOVERY_MODE POST_BUILD)
endif()
