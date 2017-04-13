/*
 170409.hpp
 Katsuki Ohto
 */

constexpr int ImageToPlains = 2;
constexpr int ImageFromPlains = 1;
constexpr int ImageDropSize = 7;
constexpr int ImageFromSize = ImageFromPlains * ImageSize + ImageDropSize;
constexpr int ImageToSize = ImageToPlains * ImageSize;
constexpr int ImageMoveOutputs = ImageFromSize + ImageToSize;

constexpr int ImageSupervisedFlags = 0;
constexpr int ImageSupervisedOutputs = ImageMoveOutputs + ImageSupervisedFlags;

void imageToMove(int from, int to, float *const p){
    *(p + from) = 1;
    *(p + ImageFromSize + to) = 1;
}

void moveToFromTo(const Move& move, const Color myColor,
                  int *const pfrom, int *const pto){
    
    // 出力データ作成
    Square sq = inverseIfWhite(myColor, move.to());
    File f = makeFile(sq);
    Rank r = makeRank(sq);
    *pto = (((int)f + ImagePadding) * ImageRankNum + (int)r + ImagePadding) * ImageToPlains;
    
    if(move.isDrop()){ // 駒打ち
        *pfrom = ImageSize + move.pieceTypeDropped() - Pawn;
    }else{
        Square sq = inverseIfWhite(myColor, move.from());
        File f = makeFile(sq);
        Rank r = makeRank(sq);
        *pfrom = ((int)f + ImagePadding) * ImageRankNum + (int)r + ImagePadding;
        
        if(move.isPromotion() || (move.pieceTypeFrom() >= ProPawn)){
            *pto += 1;
        }
    }
}