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
    auto inputTensor = tensorflow::Tensor(tensorflow::DT_FLOAT,
                                          tensorflow::TensorShape({num, ImageFileNum, ImageRankNum, ImageInputPlains}));
    auto phaseTensor = tensorflow::Tensor(tensorflow::DT_BOOL,
                                          tensorflow::TensorShape({}));
    auto mat = inputTensor.tensor<float, 4>(); // 参照
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
    phaseTensor.tensor<bool, 0>()(0) = false;
    std::vector<tensorflow::Tensor> otensors; // 出力はTensor型で受け取る
    //auto session_run_status = psession->Run({{"input:0", tensor}}, {"last/add:0"}, {}, &otensors);
    //auto session_run_status = psession->Run({{"input:0", tensor}}, {"concat:0"}, {}, &otensors);
    auto session_run_status = psession->Run({{"input:0", inputTensor}}, {"normalize/concat:0"}, {}, &otensors);
    //auto session_run_status = psession->Run({{"input:0", inputTensor}, {"phase:0", phaseTensor}}, {"concat:0"}, {}, &otensors);
    return otensors;
}

Move getBestMove(const Position& pos, bool testMode = false){
    // 状態 pos にて moves 内から最高点がついた行動を選ぶ
    ExtMove moves[1024];
    const int n = generateMoves<LegalAll>(moves, pos) - moves;
    if(n <= 0){
        if(!testMode){
            SYNCCOUT << "bestmove resign" << SYNCENDL;
        }
        return Move::moveNone();
    }
    
    BoardImage images[1];
    positionToImage(pos, pos.turn(), images[0]);
    auto otensors = forward(images, 1);
    
    // Tensor型から通常の配列型に変換
    auto mat = otensors[0].matrix<float>();
    
    // Move形式の行動それぞれの得点を計算し最高点の手を選ぶ
    //std::cerr << toOutputString(mat) << std::endl;
    
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
    if(!testMode && pos.gamePly() < 16){
        // 序盤はランダム性を入れる
        float temperature = (1 - pos.gamePly() / 16);
        float score[1024];
        float scoreSum = 0;
        for(int i = 0; i < n; ++i){
            Move m = moves[i].move;
            int from, to;
            moveToFromTo(m, pos.turn(), &from, &to);
            float tval = mat(ImageFromSize + to);
            float fval = mat(from);
            float val = fval * tval;
            if(val <= 0){
                score[i] = 0;
            }else{
                double tscore = std::exp(std::log(val) / temperature);
                score[i] = tscore;
                scoreSum += tscore;
            }
        }
        std::random_device dice;
        std::uniform_real_distribution<float> uni(0, 1);
        float r = scoreSum * uni(dice);
        int j;
        for(j = n - 1; j > 0; --j){
            r -= score[j];
            if(r < 0){
                break;
            }
        }
        bestMove = moves[j].move;
    }else{
        float bestValue = -FLT_MAX;
        for(int i = 0; i < n; ++i){
            Move m = moves[i].move;
            int from, to;
            moveToFromTo(m, pos.turn(), &from, &to);
            float tval = mat(ImageFromSize + to);
            float fval = mat(from);
            float val = fval * tval;
            //std::cerr << m.toUSI() << " " << from << " " << to << " " << val
            //<< " (" << fval << ", " << tval << ")" << std::endl;
            if(val > bestValue){
                bestMove = m;
                bestValue = val;
            }
        }
    }
    if(!testMode){
        SYNCCOUT << "bestmove " << bestMove.toUSI() << SYNCENDL;
    }
    return bestMove;
}

#ifdef LEARN
void calcAccuracy(Searcher *const psearcher,
                  const std::string& ipath){
    
    std::cerr << "accuracy test" << std::endl;
    std::cerr << "input path : " << ipath << std::endl;
    
    // 棋譜の読み込み
    Position pos(psearcher);
    Learner *plearner = new Learner();
    
    plearner->readBook(pos, ipath, "-", "-", "-", 0);
    
    if (psession == nullptr){
        // Tensorflowのセッション開始と計算グラフ読み込み
        initializeGraph("./policy_graph.pb");
    }
    
    int okCnt = 0, allCnt = 0;
    for(auto& game : plearner->bookMovesDatum_){
        pos.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                nullptr);
        std::vector<StateInfo> siv;
        for(auto& bm : game){
            siv.push_back(StateInfo());
            Move move = bm.move;
            Move tmove = getBestMove(pos, true);
            if(move == tmove){ okCnt += 1; }
            allCnt += 1;
            pos.doMove(move, siv.back());
        }
        std::cerr << okCnt / (double)allCnt << "(" << okCnt << ", " << allCnt << ")" << std::endl;
    }
}
#endif