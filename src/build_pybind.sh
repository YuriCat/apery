g++ -O3 -shared -fPIC -std=c++11 -I"$HOME/Documents/pybind11/include/" -DPYBIND11_PACKAGE -DNO_TF $(python3.6-config --cflags --ldflags) usi.cpp bitboard.cpp init.cpp mt64bit.cpp position.cpp evalList.cpp move.cpp movePicker.cpp square.cpp generateMoves.cpp evaluate.cpp search.cpp hand.cpp tt.cpp timeManager.cpp book.cpp benchmark.cpp thread.cpp common.cpp pieceScore.cpp -lpthread -o nndata.so