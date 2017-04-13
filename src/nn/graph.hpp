/*
 graph.hpp
 Katsuki Ohto
 */

// tensorflowのグラフを動かす
// 参考 http://memo.saitodev.com/home/tensorflow/use_graph_in_cxx/

#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// 共有変数
tensorflow::Session *psession = nullptr;

tensorflow::Status LoadGraph(tensorflow::string graph_filename){
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_filename, &graph_def);
    if(!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_filename, "'");
    }
    psession = tensorflow::NewSession(tensorflow::SessionOptions());
    tensorflow::Status session_create_status = psession->Create(graph_def);
    if(!session_create_status.ok()){
        return session_create_status;
    }
    return tensorflow::Status::OK();
}

int initializeGraph(const std::string& graph_filename){
    // NNの計算周りを初期化する
    // 起動時に1回だけ呼び出す
    char ***a = nullptr;
    tensorflow::port::InitMain("nn", 0, a);
    
    auto load_graph_status = LoadGraph(graph_filename);
    if(!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status.error_message();
        return -1;
    }
    return 0;
}

std::vector<tensorflow::Tensor> forward(const BoardImage images[], const int num){
    // 入力をTensor形式に変換
    auto tensor = tensorflow::Tensor(tensorflow::DT_FLOAT,
                                     tensorflow::TensorShape({num, ImageFileNum, ImageRankNum, ImageInputPlains}));
    auto mat = tensor.tensor<float, 4>(); // 参照
    mat.setZero();
    // 値を設定
    int cnt = 0;
    for(int n = 0; n < num; ++n){
        for(int i = 0; i < ImageFileNum; ++i){
            for(int j = 0; j < ImageRankNum; ++j){
                for(int p = 0; p < ImageInputPlains; ++p){
                    mat(cnt++) = float(images[n].board[i][j][p]);
                }
            }
        }
    }
    std::vector<tensorflow::Tensor> otensors; // 出力はTensor型で受け取る
    //auto session_run_status = psession->Run({{"input:0", tensor}}, {"last/add:0"}, {}, &otensors);
    auto session_run_status = psession->Run({{"input:0", tensor}}, {"concat:0"}, {}, &otensors);
    return otensors;
}

Move getBestMove(const Position& pos){
    // 状態 pos にて moves 内から最高点がついた行動を選ぶ
    ExtMove moves[1024];
    const int n = generateMoves<LegalAll>(moves, pos) - moves;
    if(n <= 0){
        SYNCCOUT << "bestmove resign" << SYNCENDL;
        return Move::moveNone();
    }
    
    BoardImage images[1];
    positionToImage(pos, pos.turn(), images[0]);
    auto otensors = forward(images, 1);
    
    // Tensor型から通常の配列型に変換
    auto mat = otensors[0].matrix<float>();
    
    // Move形式の行動それぞれの得点を計算し最高点の手を選ぶ
    /*for(int i = 0; i < ImageMoveOutputs; ++i){
     std::cerr << mat(i) << " ";
     }std::cerr << std::endl;*/
    int cnt = 0;
    for(int j = 0; j < ImageRankNum; ++j){
        for(int i = ImageFileNum - 1; i >= 0; --i){
            std::cerr << " " << std::setw(2) << int(mat(i * ImageRankNum + j) * 100);
        }
        std::cerr << std::endl;
    }std::cerr << std::endl;
    for(int i = 0; i < ImageDropSize; ++i){
        std::cerr << " " << std::setw(2) << int(mat(ImageSize + i));
    }std::cerr << std::endl;
    std::cerr << std::endl;
    for(int j = 0; j < ImageRankNum; ++j){
        for(int i = ImageFileNum - 1; i >= 0; --i){
            int ito = ImageFromSize + (i * ImageRankNum + j) * ImageToPlains;
            std::cerr << " " << std::setw(2) << int(mat(ito) * 100);
            std::cerr << "(" << std::setw(2) << int(mat(ito + 1) * 100) << ")";
        }
        std::cerr << std::endl;
    }std::cerr << std::endl;
    
    /*int from = -1, to = -1;
     float fromBestValue = -FLT_MAX, toBestVale = -FLT_MAX;
     for(int i = 0; i < BoardImage:from_size; ++i){
     float val = mat(i);
     if(val > fromBestValue){
     from = i;
     fromBestValue = val;
     }
     }
     for(int i = BoardImage::from_size; i < BoardImage::outputs; ++i){
     float val = mat(i);
     if(val > toBestValue){
     to = i;
     toBestValue = val;
     }
     }
     indexToMove(from, to);*/
    
    Move bestMove = Move::moveNone();
    float bestValue = -FLT_MAX;
    for(int i = 0; i < n; ++i){
        Move m = moves[i].move;
        int from, to;
        moveToFromTo(m, pos.turn(), &from, &to);
        float tval = mat(ImageFromSize + to);
        float fval = mat(from);
        float val = fval * tval;
        std::cerr << m.toUSI() << " " << from << " " << to << " " << val
        << " (" << fval << ", " << tval << ")" << std::endl;
        if(val > bestValue){
            bestMove = m;
            bestValue = val;
        }
    }
    SYNCCOUT << "bestmove " << bestMove.toUSI() << SYNCENDL;
    return bestMove;
}