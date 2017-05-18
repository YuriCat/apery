/*
 move170419.hpp
 Katsuki Ohto
 */

// toを駒種に分けるパターン
// 持ち駒は7ビットで別
constexpr int ImageToPlains = 14;
constexpr int ImageFromPlains = 1;
constexpr int ImageDropSize = 7;
constexpr int ImageFromSize = ImageFromPlains * ImageSize + ImageDropSize;
constexpr int ImageToSize = ImageToPlains * ImageSize;
constexpr int ImageMoveOutputs = ImageFromSize + ImageToSize;

constexpr int ImageSupervisedFlags = 0;
//constexpr int ImageSupervisedOutputs = ImageMoveOutputs + ImageSupervisedFlags;
constexpr int ImageSupervisedOutputs = 2;

void imageToMove(int from, int to, int *const p){
    //*(p + from) = 1;
    //*(p + ImageFromSize + to) = 1;
    // 配列形式で保存する必要性が無いので from と to を int で保存
    *p = from; *(p + 1) = ImageFromSize + to;
}

void moveToFromTo(const Move& move, const Color myColor,
                  int *const pfrom, int *const pto){
    
    // 出力データ作成
    Square sq = inverseIfWhite(myColor, move.to());
    File f = makeFile(sq);
    Rank r = makeRank(sq);
    *pto = (((int)f + ImagePadding) * ImageRankNum + (int)r + ImagePadding) * ImageToPlains;
    
    if(move.isDrop()){ // 駒打ち
        *pfrom = ImageSize * ImageFromPlains + move.pieceTypeDropped() - Pawn;
        *pto += move.pieceTypeDropped() - Pawn;
    }else{
        Square sq = inverseIfWhite(myColor, move.from());
        File f = makeFile(sq);
        Rank r = makeRank(sq);
        *pfrom = (((int)f + ImagePadding) * ImageRankNum + (int)r + ImagePadding) * ImageFromPlains;
        //*pfrom += move.pieceTypeFrom() - Pawn;
        
        *pto += move.pieceTypeTo() - Pawn;
    }
}

template<class array_t>
std::string toOutputString(const array_t& mat){
    std::ostringstream oss;
    /*for(int j = 0; j < ImageRankNum; ++j){
        for(int i = ImageFileNum - 1; i >= 0; --i){
            oss << " " << std::setw(2) << int(mat(i * ImageRankNum + j) * 100);
        }
        oss << std::endl;
    }oss << std::endl;
    for(int i = 0; i < ImageDropSize; ++i){
        oss << " " << std::setw(2) << int(mat(ImageSize + i));
    }oss << std::endl;
    oss << std::endl;
    for(int j = 0; j < ImageRankNum; ++j){
        for(int i = ImageFileNum - 1; i >= 0; --i){
            int ito = ImageFromSize + (i * ImageRankNum + j) * ImageToPlains;
            oss << " " << std::setw(2) << int(mat(ito) * 100);
            oss << "(" << std::setw(2) << int(mat(ito + 1) * 100) << ")";
        }
        oss << std::endl;
    }oss << std::endl;*/
    return oss.str();
}