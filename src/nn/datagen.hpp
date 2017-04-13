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
            imageToMove((*pimages)[index].from, (*pimages)[index].to, pmoveArray->data() + cnt);
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

void genPolicyTeacher(Searcher *const psearcher,
                      const std::string& ipath,
                      const std::string& opath,
                      const int threads){
    
    std::cerr << "policy teacher generation" << std::endl;
    
    std::cerr << "input path : " << ipath << std::endl;
    std::cerr << "output path : " << opath << std::endl;
    std::cerr << "threads = " << threads << std::endl;
    
    // 棋譜の読み込み
    Position pos(psearcher);
    Learner *plearner = new Learner();
    
    plearner->readBook(pos, ipath, "-", "-", "-", 0);
    
    // データをランダムに読んで教師データ作成
    u64 positionSum = 0;
    std::vector<u64> cumulativePositions;
    for(auto& game : plearner->bookMovesDatum_){
        positionSum += game.size();
        cumulativePositions.push_back(game.size());
    }
    
    std::cerr << "total positions = " << positionSum << std::endl;

    std::vector<BoardImage> images;
    images.reserve(positionSum);
    
    for(auto& game : plearner->bookMovesDatum_){
        pos.set("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                nullptr);
        std::vector<StateInfo> siv;
        for(auto& bm : game){
            siv.push_back(StateInfo());
            Color myColor = pos.turn();
            Move move = bm.move;
            pos.doMove(move, siv.back());
            //std::cerr << pos.toSFEN() << std::endl;
            
            BoardImage image;
            // 入力データ作成
            positionToImage(pos, myColor, image);
            // 出力データ作成
            moveToFromTo(move, myColor, &image.from, &image.to);
            
            images.push_back(image);
        }
        std::cerr << images.size() << std::endl;
    }
    
    // データをシャッフル
    const u32 seed = 103;//(unsigned int)time(NULL);
    std::mt19937 mt(seed);
    std::shuffle(images.begin(), images.end(), mt);
    
    // 保存
    std::vector<std::thread> savers;
    for(int th = 0; th < threads; ++th){
        savers.push_back(std::thread(&imageSaverThread, th, threads, &images, opath));
    }
    for(auto& saver : savers){
        saver.join();
    }
}