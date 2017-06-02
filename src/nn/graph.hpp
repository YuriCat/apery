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
tensorflow::Session *psession0 = nullptr, *psession1 = nullptr;

tensorflow::Status LoadGraph(tensorflow::Session **const psession,
                             const tensorflow::string& graph_filename){
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_filename, &graph_def);
    if(!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_filename, "'");
    }
    *psession = tensorflow::NewSession(tensorflow::SessionOptions());
    tensorflow::Status session_create_status = (*psession)->Create(graph_def);
    if(!session_create_status.ok()){
        return session_create_status;
    }
    return tensorflow::Status::OK();
}

int initializeGraph(tensorflow::Session **const psession,
                    const std::string& graph_filename){
    // NNの計算周りを初期化する
    // 起動時に1回だけ呼び出す
    char ***a = nullptr;
    tensorflow::port::InitMain("nn", 0, a);
    
    auto load_graph_status = LoadGraph(psession, graph_filename);
    if(!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status.error_message();
        return -1;
    }
    return 0;
}

std::vector<tensorflow::Tensor> forward(tensorflow::Session *const psession,
                                        const BoardImage images[], const int num){
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
    //phaseTensor.scalar<bool>()(0) = false;
    phaseTensor.flat<bool>().setZero();
    std::vector<tensorflow::Tensor> otensors; // 出力はTensor型で受け取る
    //auto session_run_status = psession->Run({{"input:0", tensor}}, {"last/add:0"}, {}, &otensors);
    //auto session_run_status = psession->Run({{"input:0", tensor}}, {"concat:0"}, {}, &otensors);
    //auto session_run_status = psession->Run({{"input:0", inputTensor}}, {"normalize/concat:0"}, {}, &otensors);
    std::vector<std::pair<std::string, tensorflow::Tensor>> input = {{"g/input", inputTensor}, {"g/is_training", phaseTensor}};
    
    auto session_run_status = psession->Run(input, {"g/normalize/concat"}, {}, &otensors);
    //std::cerr << session_run_status << std::endl;
    return otensors;
}

ExtMove getBestMove(const Position& pos, bool testMode = false){
    // 状態 pos にて moves 内から最高点がついた行動を選ぶ
    ExtMove moves[1024];
    const int n = generateMoves<LegalAll>(moves, pos) - moves;
    
    if(!testMode){
        if(n <= 0){
            SYNCCOUT << "bestmove resign" << SYNCENDL;
            return ExtMove(Move::moveNone(), -100000);
        }
    }
    
    BoardImage images[1];
    positionToImage(pos, pos.turn(), images[0]);
    auto otensors0 = forward(psession0, images, 1);
    auto otensors1 = forward(psession1, images, 1);
    //std::cerr << "num of tensors = " << otensors.size() << std::endl;
    
    // Tensor型から通常の配列型に変換
    auto mat = otensors0[0].matrix<float>();
    auto mat_pv = otensors1[0].matrix<float>();
    
    //std::cerr << typeid(mat).name() << std::endl;
    
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
    
    const double clipValue = 0.0000001;
    const double value = std::min(std::max(-1 + clipValue, (double)mat_pv(ImageMoveOutputs)), 1 - clipValue);
    const int score = (int)((-log((2.0 / (value + 1.0)) - 1.0) * 600) * 100 / PawnScore);
    
    //const double pref = (pos.gamePly() <= 40) ? 1.0 : std::min(1.0, 40.0 / (pos.gamePly() - 40) - );
    const double pref = (1 / (1 + (60 - pos.gamePly()) / 10.0)) * 0.9;
    
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
    }else if(pos.gamePly() < 40 || std::abs(score) > 3000){
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
    }else{
        float bestValue = -FLT_MAX;
        for(int i = 0; i < n; ++i){
            Move m = moves[i].move;
            int from, to;
            moveToFromTo(m, pos.turn(), &from, &to);
            float tval = mat(ImageFromSize + to);
            float fval = mat(from);
            float tval_pv = mat_pv(ImageFromSize + to);
            float fval_pv = mat_pv(from);
            
            float val = fval * tval * fval_pv * tval_pv;
            
            //float val = pos(fval * tval, 1 - pref) * pos(fval_pv * tval_pv, pref);
            //float val = /*fval * tval */ fval_pv * tval_pv;
            
            //float val = fval * tval;
            //float val_pv = fval_pv * tval_pv;
            
            //val = std::max(val, val_pv);
            
            //val = pow(val, 1 - pref) * pow(val_pv, pref);
            
            //std::cerr << m.toUSI() << " " << from << " " << to << " " << val
            //<< " (" << fval << ", " << tval << ")" << std::endl;
            if(val > bestValue){
                bestMove = m;
                bestValue = val;
            }
        }
    }
    
    if(!testMode){
        SYNCCOUT << "info depth 0 score cp " << score <<  " pv " << bestMove.toUSI() << SYNCENDL;
        SYNCCOUT << "bestmove " << bestMove.toUSI() << SYNCENDL;
    }
    return ExtMove(bestMove, score);
}

const std::vector<int> searchWidth = {1, 2, 4};

std::pair<std::vector<Move>, s32> searchMove(Position& pos, s32 alpha, s32 beta, int depth) {
    //std::cerr << depth << std::endl;
    if(depth <= 0){
        ExtMove bestExtMove = getBestMove(pos, true);
        std::vector<Move> pv = {bestExtMove.move};
        if(!bestExtMove.move){
            return std::make_pair(pv, -100000);
        }else{
            return std::make_pair(pv, bestExtMove.score);
        }
    }else{
        std::array<ExtMove, 1024> moves;
        const int n = generateMoves<LegalAll>(moves.data(), pos) - moves.data();
        BoardImage images[1];
        positionToImage(pos, pos.turn(), images[0]);
        auto otensors0 = forward(psession0, images, 1);
        auto otensors1 = forward(psession1, images, 1);
        auto mat = otensors0[0].matrix<float>();
        auto mat_pv = otensors1[0].matrix<float>();
        const double clipValue = 0.0000001;
        const double value = std::min(std::max(-1 + clipValue, (double)mat_pv(ImageMoveOutputs)), 1 - clipValue);
        const int score = (int)((-log((2.0 / (value + 1.0)) - 1.0) * 600) * 100 / PawnScore);
        double scoreSum = 0;
        for(int i = 0; i < n; ++i){
            Move m = moves[i].move;
            int from, to;
            moveToFromTo(m, pos.turn(), &from, &to);
            float tval;
            if(pos.gamePly() < 40 || std::abs(score) > 3000){
                tval = mat(ImageFromSize + to) * mat(from);
            }else{
                tval = mat(ImageFromSize + to) * mat(from) * mat_pv(ImageFromSize + to) * mat_pv(from);
            }
            moves[i].score = tval * 10000;
            scoreSum += tval;
        }
        // 確率正規化
        for(int i = 0; i < n; ++i){
            moves[i].score /= scoreSum;
        }
        std::sort(moves.begin(), moves.begin() + n, [](const ExtMove& a, const ExtMove& b)->bool{
            return a.score > b.score;
        });
        // 上位k手だけ探索
        std::vector<Move> pv = {Move::moveNone()};
        std::pair<std::vector<Move>, s32> bestExtPv = std::make_pair(pv, alpha);
        int bestScore = -100000;
        for(int i = 0; i < std::min(n, searchWidth[depth]); ++i){
            StateInfo si;
            pos.doMove(moves[i].move, si);
            std::pair<std::vector<Move>, s32> extPv = searchMove(pos, -beta, -alpha, depth - 1);
            const int eval = -extPv.second;
            std::cerr << std::string(3 - depth, ' ') << moves[i].move.toUSI() << " pol: " << moves[i].score / 100.0 << "% eval:" << eval << std::endl;
            int tscore = moves[i].score + 1 / (1 + exp(-eval / 600.0)) * 80000;
            if(tscore > bestScore){
                bestScore = tscore;
                extPv.first.push_back(moves[i].move);
                if(eval >= 90000){ // 詰み
                    pos.undoMove(moves[i].move);
                    return std::make_pair(extPv.first, eval);
                }
                bestExtPv = std::make_pair(extPv.first, (score + eval * depth) / (depth + 1));
            }
            alpha = std::max((score + eval * depth) / (depth + 1), alpha);
            pos.undoMove(moves[i].move);
        }
        return bestExtPv;
    }
}

std::string toPvString(const std::vector<Move>& pv){
    std::ostringstream oss;
    for(int i = (int)pv.size() - 1; i >= 0; --i){
        if(pv[i]){
            oss << " " << pv[i].toUSI();
        }
    }
    return oss.str();
}

std::pair<std::vector<Move>, s32> getBestSearchMove(Position& pos){
    std::pair<std::vector<Move>, s32> bestExtPv = searchMove(pos, -100000, 100000, 2);
    std::cerr << bestExtPv.second << std::endl;
    if(bestExtPv.second <= -100000){
        SYNCCOUT << "bestmove resign" << SYNCENDL;
    }else{
        SYNCCOUT << "info depth " << (bestExtPv.first.size() - 1) << " score cp " << bestExtPv.second <<  " pv" << toPvString(bestExtPv.first) << SYNCENDL;
        std::cerr << "bestmove " << bestExtPv.first.back().toUSI() << std::endl;
        SYNCCOUT << "bestmove " << bestExtPv.first.back().toUSI() << SYNCENDL;
    }
    return bestExtPv;
}

template<class callback_t>
void calcMoveProb(const Position& pos, const callback_t& callback){
    // 状態 pos にて moves 内 の行動集合に選択確率をつける
    BoardImage images[1];
    positionToImage(pos, pos.turn(), images[0]);
    auto otensors = forward(psession0, images, 1);
    //std::cerr << "num of tensors = " << otensors.size() << std::endl;
    
    // Tensor型から通常の配列型に変換
    auto mat = otensors[0].matrix<float>();
    
    callback(mat);
}

#ifdef LEARN
/*void calcAccuracy(Searcher *const psearcher,
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
        std::deque<StateInfo> siv;
        for(auto& bm : game){
            Move move = bm.move;
            Move tmove = getBestMove(pos, true);
            if(move == tmove){ okCnt += 1; }
            allCnt += 1;
            siv.push_back(StateInfo());
            pos.doMove(move, siv.back());
        }
        std::cerr << okCnt / (double)allCnt << "(" << okCnt << ", " << allCnt << ")" << std::endl;
    }
}*/

void calcAccuracy(Searcher *const psearcher,
                  const std::string& ipath){
    
    std::cerr << "accuracy test" << std::endl;
    std::cerr << "input path : " << ipath << std::endl;
    
    const int trainPositionSum = 16300 * 1024;
    const int testPositionSum = 128 * 1024;
    
    const u32 seed = 103;
    std::mt19937 mt(seed);
    
    // 棋譜の読み込み
    Position pos(psearcher);
    Learner *plearner = new Learner();
    
    plearner->readBook(pos, ipath, "-", "-", "-", 0);
    
    if (psession0 == nullptr){
        // Tensorflowのセッション開始と計算グラフ読み込み
        initializeGraph(&psession0, "./policy_graph.pb");
    }
    
    u64 positionSum = 0;
    for(auto& game : plearner->bookMovesDatum_){
        positionSum += game.size();
    }
    
    // データを読む順番を決定
    std::vector<int> order;
    order.reserve(positionSum);
    for(int i = 0; i < (int)positionSum; ++i){
        order.push_back(i);
    }
    std::shuffle(order.begin(), order.end(), mt);

    
    std::vector<std::tuple<Position, Move, int>> positions;
    positions.reserve(positionSum);
    
    for(auto& game : plearner->bookMovesDatum_){
        pos.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                nullptr);
        std::deque<StateInfo> siv;
        int ply = 0;
        for(auto& bm : game){
            Move move = bm.move;
            positions.push_back(std::make_tuple(pos, move, ply));
            siv.push_back(StateInfo());
            pos.doMove(move, siv.back());
            ++ply;
        }
    }
    std::shuffle(positions.begin(), positions.end(), mt);
    
    // for test data
    std::cerr << "for test data" << std::endl;
    {
        int okCnt = 0, allCnt = 0, okPlyCnt[6] = {0}, allPlyCnt[6] = {0};
        for(int i = 0; i < testPositionSum; ++i){
            int index = trainPositionSum + i;
            const Position& tpos = std::get<0>(positions[index]);
            const Move move = std::get<1>(positions[index]);
            const int ply = std::get<2>(positions[index]);
            Move tmove = getBestMove(tpos, true).move;
            int phase = std::min(5, ply / 30);
            if(move == tmove){
                okCnt += 1;
                okPlyCnt[phase] += 1;
            }
            allCnt += 1;
            allPlyCnt[phase] += 1;
            if(i % 1000 == 999){
                std::cerr << okCnt / (double)allCnt << "(" << okCnt << " / " << allCnt << ")" << std::endl;
                for(int ph = 0; ph < 6; ++ph){
                    std::cerr << "[" << ph * 30  << " ~] : ";
                    std::cerr << okPlyCnt[ph] / (double)allPlyCnt[ph] << "(" << okPlyCnt[ph] << " / " << allPlyCnt[ph] << ")" << std::endl;
                }
            }
        }
        std::cerr << okCnt / (double)allCnt << "(" << okCnt << " / " << allCnt << ")" << std::endl;
        for(int ph = 0; ph < 6; ++ph){
            std::cerr << okPlyCnt[ph] / (double)allPlyCnt[ph] << "(" << okPlyCnt[ph] << " / " << allPlyCnt[ph] << ")" << std::endl;
        }
    }
    // for training data
    std::cerr << "for train data" << std::endl;
    {
        int okCnt = 0, allCnt = 0, okPlyCnt[6] = {0}, allPlyCnt[6] = {0};
        for(int i = 0; i < testPositionSum; ++i){
            int index = trainPositionSum - testPositionSum + i;
            const Position& tpos = std::get<0>(positions[index]);
            const Move move = std::get<1>(positions[index]);
            const int ply = std::get<2>(positions[index]);
            Move tmove = getBestMove(tpos, true).move;
            int phase = std::min(5, ply / 30);
            if(move == tmove){
                okCnt += 1;
                okPlyCnt[phase] += 1;
            }
            allCnt += 1;
            allPlyCnt[phase] += 1;
            if(i % 1000 == 999){
                std::cerr << okCnt / (double)allCnt << "(" << okCnt << " / " << allCnt << ")" << std::endl;
                for(int ph = 0; ph < 6; ++ph){
                    std::cerr << "[" << ph * 30  << " ~] : ";
                    std::cerr << okPlyCnt[ph] / (double)allPlyCnt[ph] << "(" << okPlyCnt[ph] << " / " << allPlyCnt[ph] << ")" << std::endl;
                }
            }
        }
        std::cerr << okCnt / (double)allCnt << "(" << okCnt << " / " << allCnt << ")" << std::endl;
        for(int ph = 0; ph < 6; ++ph){
            std::cerr << okPlyCnt[ph] / (double)allPlyCnt[ph] << "(" << okPlyCnt[ph] << " / " << allPlyCnt[ph] << ")" << std::endl;
        }
    }
}

#endif