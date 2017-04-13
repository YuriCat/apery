/*
 input170405.hpp
 Katsuki Ohto
 */

#include <bitset>

constexpr int ImageInputPlains = 107;
constexpr int ImageInputs = ImageInputPlains * ImageSize;

struct BoardImage{
    // NNへのインプットデータ
    std::bitset<ImageInputPlains> board[ImageFileNum][ImageRankNum];
    
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
        for(int i = 0; i < ImageFileNum; ++i){
            for(int j = 0; j < ImageRankNum; ++j){
                board[i][j].reset();
            }
        }
        from = -1;
        to = -1;
    }
    
    void fill(int index){
        for(int i = 0; i < ImageFileNum; ++i){
            for(int j = 0; j < ImageRankNum; ++j){
                board[i][j].set(index);
            }
        }
    }
    
    int from, to;
};

void imageToInput(const BoardImage& image, float *pi){
    // ニューラルネットの入力として使用する型への変換
    for(int i = 0; i < ImageFileNum; ++i){
        for(int j = 0; j < ImageRankNum; ++j){
            for(int p = 0; p < ImageInputPlains; ++p){
                (*pi) = float(image.board[i][j][p]);
                ++pi;
            }
        }
    }
}

void positionToImage(const Position& pos, const Color myColor, BoardImage& image){
    image.clear();
    
    // 盤内
    for(int i = 0; i < FileNum; ++i){
        for(int j = 0; j < RankNum; ++j){
            image.board[i + ImagePadding][j + ImagePadding].set(0);
        }
    }
    
    // 盤内の駒
    for(int i = 0; i < FileNum; ++i){
        for(int j = 0; j < RankNum; ++j){
            // 盤上の位置の計算
            // 試合開始時点での後手が手盤を持つ時には盤面を反転させる
            Square sq = inverseIfWhite(myColor, makeSquare(File(i), Rank(j)));
            Piece p = pos.piece(sq);
            if(p != Empty){
                Color pc = pieceToColor(p);
                PieceType pt = pieceToPieceType(p);
                
                if(pc == myColor){ // 手番側
                    image.board[i + ImagePadding][j + ImagePadding].set(pt - Pawn + 1);
                }else{
                    image.board[i + ImagePadding][j + ImagePadding].set(pt - Pawn + 15);
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
}