/*
 170409.hpp
 Katsuki Ohto
 */

#include <bitset>
#include "../cnpy.h"

struct BoardImage{
    // NNへのインプットデータ
    static constexpr int plains = 107;
    
    std::bitset<plains> board[11][11];
    
    // 0 盤内に1
    // 1  ~ 14 先手の駒
    // 15 ~ 28 後手の駒
    // 29 ~ 66 先手持ち駒
    // 67 ~ 106 後手持ち駒
    
    // 以降未実装
    // 65 ~ 78 先手の駒の現実の効き
    // 79 ~ 92 後手の駒の現実の効き
    // 93 ~ 106 先手の駒の理想的な効き
    // 107 ~ 120 後手の駒の理想的な効き
    // 121 先手の成り
    // 122 後手の成り
    // 123 先手の歩のある筋
    // 124 後手の歩のある筋
    // 125 ~ 137 後手玉に対する先手の王手位置
    // 138 ~ 150 先手玉に対する後手の王手位置
    
    void clear(){
        for(int i = 0; i < 11; ++i){
            for(int j = 0; j < 11; ++j){
                board[i][j].reset();
            }
        }
        from = -1;
        to = -1;
        promote = -1;
    }
    
    void fill(int index){
        for(int i = 0; i < 11; ++i){
            for(int j = 0; j < 11; ++j){
                board[i][j].set(index);
            }
        }
    }
    
    // NNからのアウトプットデータ
    int from, to;
    int promote;
};

std::ostream& operator <<(std::ostream& ost, const BoardImage& bi){
    ost << bi.from << ' ' << bi.to << ' ' << bi.promote;
    for(int i = 0; i < 11; ++i){
        for(int j = 0; j < 11; ++j){
            ost << ' ' << bi.board[i][j];
        }
    }
    return ost;
}

void imageSaverThread(const int threadIndex, const int threads,
                      const std::vector<BoardImage> *const pimages,
                      const std::string& opath){
    
    constexpr int batchSize = 1024;
    const int fileNum = pimages->size() / batchSize;
    
    // 学習データ保存をスレッドで分担して行う
    // .npz形式での保存
    auto *const pinputArray = new std::array<float, batchSize * 11 * 11 * BoardImage::plains>();
    auto *const pmoveArray = new std::array<float, batchSize * (11 * 11 + 7 + 11 * 11 * 2)>();
    const std::vector<unsigned int> inputShape = {batchSize, 11, 11, BoardImage::plains};
    const std::vector<unsigned int> moveShape = {batchSize, 11 * 11 + 7 + 11 * 11 * 2};
    
    for(int fileIndex = threadIndex; fileIndex < fileNum; fileIndex += threads){
        int cnt;
        // input
        const std::string fileName = opath + std::to_string(fileIndex) + ".npz";
        cnt = 0;
        for(int dataIndex = 0; dataIndex < batchSize; ++dataIndex){
            int index = fileIndex * batchSize + dataIndex;
            for(int i = 0; i < 11; ++i){
                for(int j = 0; j < 11; ++j){
                    for(int p = 0; p < BoardImage::plains; ++p){
                        (*pinputArray)[cnt++] = (*pimages)[index].board[i][j][p];
                    }
                }
            }
        }
        // from
        pmoveArray->fill(0);
        int mcnt = 0;
        for(int dataIndex = 0; dataIndex < batchSize; ++dataIndex){
            int index = fileIndex * batchSize + dataIndex;
            const int from = (*pimages)[index].from;
            const int to = (*pimages)[index].to;
            const int promote = (*pimages)[index].promote;
            (*pmoveArray)[mcnt + from] = 1;
            (*pmoveArray)[mcnt + 11 * 11 + 7 + to * 2 + promote] = 1;
            mcnt += 11 * 11 + 7 + 11 * 11 * 2;
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
            Move move = bm.move;
            pos.doMove(move, siv.back());
            //std::cerr << pos.toSFEN() << std::endl;
            
            // 入力データ作成
            BoardImage image;
            
            image.clear();
            
            // 盤内
            for(int i = 0; i < 9; ++i){
                for(int j = 0; j < 9; ++j){
                    image.board[i + 1][j + 1].set(0);
                }
            }
            
            // 盤内の駒
            Color myColor = pos.turn();
            for(int i = 0; i < 9; ++i){
                for(int j = 0; j < 9; ++j){
                    // 盤上の位置の計算
                    // 試合開始時点での後手が手盤を持つ時には盤面を反転させる
                    Square sq = inverseIfWhite(myColor, makeSquare(File(i), Rank(j)));
                    Piece p = pos.piece(sq);
                    if(p != Empty){
                        Color pc = pieceToColor(p);
                        PieceType pt = pieceToPieceType(p);
                        
                        if(pc == myColor){ // 手番側
                            image.board[i + 1][j + 1].set(pt - Pawn + 1);
                        }else{
                            image.board[i + 1][j + 1].set(pt - Pawn + 15);
                        }
                    }
                }
            }
            
            // 持ち駒
            // 手番側
            Hand myHand = pos.hand(myColor);
            for(int i = 0; i < (int)myHand.numOf<HPawn>(); ++i){
                image.fill(29 + i);
            }
            for(int i = 0; i < (int)myHand.numOf<HLance>(); ++i){
                image.fill(29 + 18 + i);
            }
            for(int i = 0; i < (int)myHand.numOf<HKnight>(); ++i){
                image.fill(29 + 18 + 4 + i);
            }
            for(int i = 0; i < (int)myHand.numOf<HSilver>(); ++i){
                image.fill(29 + 18 + 4 + 4 + i);
            }
            for(int i = 0; i < (int)myHand.numOf<HGold>(); ++i){
                image.fill(29 + 18 + 4 + 4 + 4 + i);
            }
            for(int i = 0; i < (int)myHand.numOf<HBishop>(); ++i){
                image.fill(29 + 18 + 4 + 4 + 4 + 4 + i);
            }
            for(int i = 0; i < (int)myHand.numOf<HRook>(); ++i){
                image.fill(29 + 18 + 4 + 4 + 4 + 4 + 2 + i);
            }
            // 相手側
            Hand oppHand = pos.hand(oppositeColor(myColor));
            for(int i = 0; i < (int)oppHand.numOf<HPawn>(); ++i){
                image.fill(67 + i);
            }
            for(int i = 0; i < (int)oppHand.numOf<HLance>(); ++i){
                image.fill(67 + 18 + i);
            }
            for(int i = 0; i < (int)oppHand.numOf<HKnight>(); ++i){
                image.fill(67 + 18 + 4 + i);
            }
            for(int i = 0; i < (int)oppHand.numOf<HSilver>(); ++i){
                image.fill(67 + 18 + 4 + 4 + i);
            }
            for(int i = 0; i < (int)oppHand.numOf<HGold>(); ++i){
                image.fill(67 + 18 + 4 + 4 + 4 + i);
            }
            for(int i = 0; i < (int)oppHand.numOf<HBishop>(); ++i){
                image.fill(67 + 18 + 4 + 4 + 4 + 4 + i);
            }
            for(int i = 0; i < (int)oppHand.numOf<HRook>(); ++i){
                image.fill(67 + 18 + 4 + 4 + 4 + 4 + 2 + i);
            }
            
            // 出力データ作成
            if(move.isDrop()){ // 駒打ち
                image.from = 11 * 11 + move.pieceTypeDropped() - Pawn;
                image.promote = 0;
            }else{
                Square sq = inverseIfWhite(myColor, move.from());
                File f = makeFile(sq);
                Rank r = makeRank(sq);
                image.from = ((int)f + 1) * 11 + (int)r + 1;
                
                image.promote = (move.isPromotion() || (move.pieceTypeFrom() >= ProPawn)) ? 1 : 0;
            }
            
            Square sq = inverseIfWhite(myColor, move.to());
            File f = makeFile(sq);
            Rank r = makeRank(sq);
            image.to = ((int)f + 1) * 11 + (int)r + 1;
            
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