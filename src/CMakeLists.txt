set(ALPHAZERO_ENGINE_SOURCES
    games/chess/chess_state.cpp
    games/chess/bitboard.cpp
    games/chess/movegen.cpp
    games/chess/chess_config.cpp
    games/go/go_state.cpp
    games/go/go_rules.cpp
    games/go/scoring.cpp
    games/go/go_config.cpp
    mcts/arena_node_store.cpp
    mcts/mcts_search.cpp
    mcts/eval_queue.cpp
    selfplay/self_play_manager.cpp
    selfplay/self_play_game.cpp
    selfplay/replay_buffer.cpp
    nn/libtorch_inference.cpp
)

add_library(alphazero_engine STATIC ${ALPHAZERO_ENGINE_SOURCES})

target_compile_features(alphazero_engine PUBLIC cxx_std_20)

target_include_directories(alphazero_engine
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)

if(MSVC)
    target_compile_options(alphazero_engine PRIVATE /W4)
else()
    target_compile_options(alphazero_engine PRIVATE -Wall -Wextra)
endif()

find_package(Threads REQUIRED)
target_link_libraries(alphazero_engine PUBLIC Threads::Threads)

if(ALPHAZERO_HAS_TORCH)
    target_compile_definitions(alphazero_engine PUBLIC ALPHAZERO_HAS_TORCH=1)
    if(TORCH_INCLUDE_DIRS)
        target_include_directories(alphazero_engine PUBLIC ${TORCH_INCLUDE_DIRS})
    endif()
    if(TORCH_CXX_FLAGS)
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(alphazero_engine PUBLIC ${TORCH_CXX_FLAGS_LIST})
    endif()
    target_link_libraries(alphazero_engine PUBLIC ${TORCH_LIBRARIES})
else()
    target_compile_definitions(alphazero_engine PUBLIC ALPHAZERO_HAS_TORCH=0)
endif()

if(ALPHAZERO_HAS_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(alphazero_engine PUBLIC CUDA::cudart)
endif()

if(ALPHAZERO_HAS_PYBIND)
    pybind11_add_module(alphazero_cpp bindings/python_bindings.cpp)
    target_compile_features(alphazero_cpp PRIVATE cxx_std_20)
    target_link_libraries(alphazero_cpp PRIVATE alphazero_engine)
endif()
