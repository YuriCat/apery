/*
 datagen.hpp
 Katsuki Ohto
 */

#include "../cnpy.h"

void imageSaverThread(const int threadIndex, const int threads,
                      const std::vector<BoardImage> *const pimages,
                      const std::string& opath){
    
    constexpr int batchSize = 1024;
    const int fileNum = pimages->size() / batchSize;
    
    // 学習データ保存をスレッドで分担して行う
    // .npz形式での保存
    auto *const pinputArray = new std::array<float, batchSize * ImageInputs>();
    auto *const pmoveArray = new std::array<float, batchSize * ImageSupervisedOutputs>();
    const std::vector<unsigned int> inputShape = {batchSize, ImageFileNum, ImageRankNum, ImageInputPlains};
    const std::vector<unsigned int> moveShape = {batchSize, ImageSupervisedOutputs};
    
    for(int fileIndex = threadIndex; fileIndex < fileNum; fileIndex += threads){
        int cnt;
        // input
        const std::string fileName = opath + std::to_string(fileIndex) + ".npz";
        cnt = 0;
        for(int dataIndex = 0; dataIndex < batchSize; ++dataIndex){
            int index = fileIndex * batchSize + dataIndex;
            imageToInput((*pimages)[index], pinputArray->data() + cnt);
            cnt += ImageInputs;
        }
        // from
        pmoveArray->fill(0);
        int mcnt = 0;
        for(int dataIndex = 0; dataIndex < batchSize; ++dataIndex){
            int index = fileIndex * batchSize + dataIndex;
            imageToMove((*pimages)[index].from, (*pimages)[index].to, pmoveArray->data() + mcnt);
            mcnt += ImageSupervisedOutputs;
        }
        std::cerr << fileName << std::endl;
        cnpy::npz_save(fileName, "input",
                       pinputArray->data(), inputShape.data(), inputShape.size(), "w");
        cnpy::npz_save(fileName, "move",
                       pmoveArray->data(), moveShape.data(), moveShape.size(), "a");
    }
    
    delete pinputArray;
    delete pmoveArray;
}

void imageGeneratorThread(const int threadIndex, const int threads,
                          const int fileNum, // 全体で何個データを作成するか
                          const std::vector<int> *const porder, // データの作成順指示
                          const std::vector<u64> *const pcumulativePositions, // 累積教師局面数
                          const std::vector<std::vector<BookMoveData>> *const precords, // 試合棋譜
                          Searcher *const psearcher, // これがないとPositionが作れない
                          const std::string& opath){
    
    constexpr int batchSize = 1024;
    
    Position pos(psearcher);
    
    // 学習データの作成と保存をスレッドで分担して行う
    // .npz形式での保存
    auto *const pinputArray = new std::array<float, batchSize * ImageInputs>();
    auto *const pmoveArray = new std::array<float, batchSize * ImageSupervisedOutputs>();
    const std::vector<unsigned int> inputShape = {batchSize, ImageFileNum, ImageRankNum, ImageInputPlains};
    const std::vector<unsigned int> moveShape = {batchSize, ImageSupervisedOutputs};
    
    for(int fileIndex = threadIndex; fileIndex < fileNum; fileIndex += threads){
        const std::string fileName = opath + std::to_string(fileIndex) + ".npz";
        for(int dataIndex = 0; dataIndex < batchSize; ++dataIndex){
            const int index = (*porder)[fileIndex * batchSize + dataIndex];
            pinputArray->fill(0);
            pmoveArray->fill(0);
            
            int icnt = 0, mcnt = 0;
            // 目的のインデックスのデータまで進める
            int gameIndex = std::upper_bound(pcumulativePositions->begin(),
                                             pcumulativePositions->end(),
                                             index) - pcumulativePositions->begin() - 1;
            int ply = index - (*pcumulativePositions)[gameIndex];
            const std::vector<BookMoveData>& game = (*precords)[gameIndex];
            pos.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", nullptr);
            std::deque<StateInfo> siv;
            for(int i = 0; i < ply; ++i){
                siv.push_back(StateInfo());
                pos.doMove(game[i].move, siv.back());
            }
            const Color myColor = pos.turn();
            const Move move = game[ply].move;
            BoardImage image;
            // inputデータ作成
            positionToImage(pos, myColor, image);
            // outputデータ作成
            moveToFromTo(move, myColor, &image.from, &image.to);
            // input
            imageToInput(image, pinputArray->data() + icnt);
            icnt += ImageInputs;
            // move
            imageToMove(image.from, image.to, pmoveArray->data() + mcnt);
            mcnt += ImageSupervisedOutputs;
        }
        std::cerr << fileName << std::endl; // 保存時に落ちた場合に気づくように先にファイル名出力
        cnpy::npz_save(fileName, "input",
                       pinputArray->data(), inputShape.data(), inputShape.size(), "w");
        cnpy::npz_save(fileName, "move",
                       pmoveArray->data(), moveShape.data(), moveShape.size(), "a");
    }
    delete pinputArray;
    delete pmoveArray;
}

#include <unordered_map>

void k(){
    int t[100000000000000000];
    std::cerr << t[1000000000000] << std::endl;
}

bool isInverse(const BoardImage& img0, const BoardImage& img1){
    for(int k = 0; k < ImageInputPlains; ++k){
        for(int i = 0; i < ImageFileNum; ++i){
            for(int j = 0; j < ImageRankNum; ++j){
                if(img0.board[i][j][k] != img1.board[i][j][k]){
                    std::cerr << i << ", " << j << " : " << k << " ";
                    std::cerr << img0.board[i][j][k] << " <-> " << img1.board[i][j][k] << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

void genPolicyTeacher(Searcher *const psearcher,
                      const std::string& ipath,
                      const std::string& opath,
                      const int threads){
    
    Bitboard bb(413236765456111111 + (1 << 20), 1321);
    Bitboard cc(inverse(bb));
    
    bb.printBoard();
    cc.printBoard();
    
    /*std::vector<std::thread> thr;
    for(int th = 0; th < threads; ++th){
        thr.push_back(std::thread(&k));
    }*/
    
    std::cerr << "policy teacher generation" << std::endl;
    
    std::cerr << "input path : " << ipath << std::endl;
    std::cerr << "output path : " << opath << std::endl;
    std::cerr << "threads = " << threads << std::endl;
    
    // シャッフル用のサイコロ用意
    const u32 seed = 103;
    //const u32 seed = (unsigned int)time(NULL);
    std::mt19937 mt(seed);
    
    {
        // 対称性チェック
        Position pos(psearcher);
        pos.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                nullptr);
        BoardImage img0, img1;
        positionToImage(pos, Black, img0);
        positionToImage(pos, White, img1);
        //std::cerr << pos.hand(Black).value() << " " << pos.hand(White).value() << std::endl;
        std::cerr << "inverse check - ";
        std::cerr << (isInverse(img0, img1) ? "OK" : "Failed") << std::endl;
    }
    
    // 棋譜の読み込み
    Position pos(psearcher);
    Learner *plearner = new Learner();
    
    plearner->readBook(pos, ipath, "-", "-", "-", 0);
    
    // 局面頻度表の計算
    /*std::unordered_map<u64, int> hash;
    
    for(auto& game : plearner->bookMovesDatum_){
        pos.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                nullptr);
        std::vector<StateInfo> siv;
        for(auto& bm : game){
            hash[pos.getKey()] += 1;
            siv.push_back(StateInfo());
            Move move = bm.move;
            pos.doMove(move, siv.back());
        }
    }
    
    std::vector<int> v;
    for(auto& d : hash){
        v.push_back(d.second);
    }
    std::sort(v.begin(), v.end(), std::greater<int>());
    
    std::ofstream ofs("./state.csv");
    for(int i = 0; i < v.size(); ++i){
        ofs << (i + 1) << "," << v[i] << "," << log(i + 1) << "," << log(v[i]) << std::endl;
    }
    ofs.close();
    
    exit(1);*/
    
    //int handNum[]
    
    // データをランダムに読んで教師データ作成
    u64 positionSum = 0;
    std::vector<u64> cumulativePositions;
    for(auto& game : plearner->bookMovesDatum_){
        positionSum += game.size();
        cumulativePositions.push_back(game.size());
    }
    
    std::cerr << "total positions = " << positionSum << std::endl;

    // データを読む順番を決定
    std::vector<int> order;
    order.reserve(positionSum);
    for(int i = 0; i < positionSum; ++i){
        order.push_back(i);
    }
    std::shuffle(order.begin(), order.end(), mt);
    
    std::vector<BoardImage> images;
    images.reserve(positionSum);
    
    for(auto& game : plearner->bookMovesDatum_){
        pos.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                nullptr);
        std::deque<StateInfo> siv;
        for(auto& bm : game){
            siv.push_back(StateInfo());
            Color myColor = pos.turn();
            Move move = bm.move;
            //std::cerr << pos.toSFEN() << std::endl;
            
            BoardImage image;
            // 入力データ作成
            positionToImage(pos, myColor, image);
            // 出力データ作成
            moveToFromTo(move, myColor, &image.from, &image.to);
            
            //std::cerr << myColor << " " << move.toUSI() << " (" << image.from << ", " << image.to << ")" << std::endl;
            //getBestMove(
            
            images.push_back(image);
                        
            pos.doMove(move, siv.back());
        }
        std::cerr << images.size() << std::endl;
    }
    
    std::shuffle(images.begin(), images.end(), mt);
    
    // 保存
    std::vector<std::thread> savers;
    for(int th = 0; th < threads; ++th){
        savers.push_back(std::thread(&imageSaverThread, th, threads, &images, opath));
    }
    for(auto& saver : savers){
        saver.join();
    }
    
    // 各スレッドに作成と保存を投げる
    /*std::vector<std::thread> generators;
    for(int th = 0; th < threads; ++th){
        generators.push_back(std::thread(&imageGeneratorThread, th, threads, positionSum,
                                         &order, &cumulativePositions, &plearner->bookMovesDatum_,
                                         psearcher, opath));
    }
    for(auto& generator : generators){
        generator.join();
    }*/
}