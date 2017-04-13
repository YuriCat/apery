/*
 input170413.hpp
 Katsuki Ohto
 */

#include <bitset>

constexpr int ImageInputPlains = 119;
constexpr int ImageInputs = ImageInputPlains * ImageSize;

u32 reverseBits(u32 v){
    v = (v >> 16) | (v << 16);
    v = ((v >> 8) & 0x00ff00ff) | ((v & 0x00ff00ff) << 8);
    v = ((v >> 4) & 0x0f0f0f0f) | ((v & 0x0f0f0f0f) << 4);
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    return ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
}
u64 reverseBits(u64 v)noexcept{
    v = (v >> 32) | (v << 32);
    v = ((v >> 16) & 0x0000ffff0000ffff) | ((v & 0x0000ffff0000ffff) << 16);
    v = ((v >> 8) & 0x00ff00ff00ff00ff) | ((v & 0x00ff00ff00ff00ff) << 8);
    v = ((v >> 4) & 0x0f0f0f0f0f0f0f0f) | ((v & 0x0f0f0f0f0f0f0f0f) << 4);
    v = ((v >> 2) & 0x3333333333333333) | ((v & 0x3333333333333333) << 2);
    return  ((v >> 1) & 0x5555555555555555) | ((v & 0x5555555555555555) << 1);
}

Bitboard inverse(Bitboard bb){
    u64 b0 = bb.p(0);
    u32 b1 = u32(bb.p(1));
    u64 rb0 = reverseBits(b0);
    u32 rb1 = reverseBits(b1);
    
    u64 nb1= rb0 >> (64 - RankNum * 2);
    u64 nb0 = (rb1 >> (32 - RankNum * 2)) | (rb0 << (SquareNum - 64));
    nb0 &= 0xEFFFFFFFFFFFFFFF;
    return Bitboard(nb0, nb1);
}
Bitboard inverseIfWhite(Color c, Bitboard bb){
    return (c == Black) ? bb : inverse(bb);
}

struct BoardImage{
    // NNへのインプットデータ
    std::bitset<ImageInputPlains> board[ImageFileNum][ImageRankNum];
    
    // 0 盤内に1
    // 1  ~ 14 先手の駒
    // 15 ~ 28 後手の駒
    // 29 ~ 66 先手持ち駒
    // 67 ~ 106 後手持ち駒
    
    // 以降未実装
    // 107 先手の成りゾーン
    // 108 後手の成りゾーン

    // 109 dcBB
    // 110 ピンされている駒
    
    // 111 先手の歩のある筋
    // 112 後手の歩のある筋
    
    // 113 ~ 115 先手の駒の利きの数
    // 116 ~ 118 後手の駒の利きの数
    
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
    
    void fillRank(int index, int j){
        for(int i = 0; i < ImageFileNum; ++i){
            board[i][j].set(index);
        }
    }
    void fillFile(int index, int i){
        for(int j = 0; j < ImageRankNum; ++j){
            board[i][j].set(index);
        }
    }
    
    void fillBB(int index, const Bitboard& bb){
        for(int i = 0; i < FileNum; ++i){
            for(int j = 0; j < RankNum; ++j){
                if(bb.isSet(makeSquare(File(i), Rank(j)))){
                    board[i + ImagePadding][j + ImagePadding].set(index);
                }
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
    
    // 成りゾーン
    for(int i = 0; i < FileNum; ++i){
        for(int j = 0; j < 3; ++j){
            image.board[i + ImagePadding][j + ImagePadding].set(107);
        }
    }
    for(int i = 0; i < FileNum; ++i){
        for(int j = RankNum - 3; j < RankNum; ++j){
            image.board[i + ImagePadding][j + ImagePadding].set(108);
        }
    }
    
    // 王手情報
    CheckInfo checkInfo(pos);
    image.fillBB(109, inverseIfWhite(myColor, checkInfo.dcBB));
    image.fillBB(110, inverseIfWhite(myColor, checkInfo.pinned));
    
    // 歩のある筋
    for(int i = 0; i < FileNum; ++i){
        for(int j = 0; j < RankNum; ++j){
            Square sq = inverseIfWhite(myColor, makeSquare(File(i), Rank(j)));
            Piece p = pos.piece(sq);
            if(pieceToPieceType(p) == Pawn){
                image.fillFile(111 + ((pieceToColor(p) != myColor) ? 1 : 0), i);
            }
        }
    }
    
    // 利き
    Bitboard occ = inverseIfWhite(myColor, pos.occupiedBB());
    for(int i = 0; i < FileNum; ++i){
        for(int j = 0; j < RankNum; ++j){
            Square sq = inverseIfWhite(myColor, makeSquare(File(i), Rank(j)));
            Bitboard opponentAttackers = pos.attackersTo(myColor, sq, occ);
            for(int k = 1; k < std::min(3, opponentAttackers.popCount()); ++k){
                image.board[i + ImagePadding][j + ImagePadding].set(113 - 1 + k);
            }
            Bitboard myAttackers = pos.attackersTo(oppositeColor(myColor), sq, occ);
            for(int k = 1; k < std::min(3, myAttackers.popCount()); ++k){
                image.board[i + ImagePadding][j + ImagePadding].set(116 - 1 + k);
            }
        }
    }
}