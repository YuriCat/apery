/*
 move170405.hpp
 Katsuki Ohto
 */

constexpr int ImageToPlains = 9;
constexpr int ImageFromPlains = 1;
constexpr int ImageMovePlains = ImageFromPlains + ImageToPlains;
constexpr int ImageFromSize = ImageFromPlains * ImageSize;
constexpr int ImageToSize = ImageToPlains * ImageSize;
constexpr int ImageMoveOutputs = ImageFromSize + ImageToSize;

constexpr int ImageSupervisedFlags = 1;
constexpr int ImageSupervisedOutputs = ImageMoveOutputs + ImageSupervisedFlags;

void imageToMove(int from, int to, float *const p){
    *(p + ImageFromSize + to) = 1;
    if(to >= ImageSize * 2){ // 駒打ち
        *(p + ImageMoveOutputs) = 1;
    }else{
        *(p + from) = 1;
    }
}

void moveToFromTo(const Move& move, const Color myColor,
                  int *const pfrom, int *const pto){
    // 出力データ作成
    Square sq = inverseIfWhite(myColor, move.to());
    File f = makeFile(sq);
    Rank r = makeRank(sq);
    *pto = (((int)f + ImagePadding) * ImageRankNum + (int)r + ImagePadding) * ImageToPlains;
    
    if(move.isDrop()){ // 駒打ち
        *pfrom = -1; // どうでもよい
        *pto += move.pieceTypeDropped() - Pawn + 2;
    }else{
        Square sq = inverseIfWhite(myColor, move.from());
        File f = makeFile(sq);
        Rank r = makeRank(sq);
        *pfrom = ((int)f + ImagePadding) * ImageRankNum + (int)r + ImagePadding;
        
        if(move.isPromotion()){
            *pto += 1;
        };
    }
}