/*
  Apery, a USI shogi playing engine derived from Stockfish, a UCI chess playing engine.
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2016 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad
  Copyright (C) 2011-2016 Hiraoka Takuya

  Apery is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Apery is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "usi.hpp"
#include "position.hpp"
#include "move.hpp"
#include "movePicker.hpp"
#include "generateMoves.hpp"
#include "search.hpp"
#include "tt.hpp"
#include "book.hpp"
#include "thread.hpp"
#include "benchmark.hpp"
#include "learner.hpp"

#include "init.hpp"

// 以下NN関係
//#include "nn/def170405.hpp"
#include "nn/def170519.hpp"
//#include "nn/input170405.hpp"
//#include "nn/input170413.hpp"
#include "nn/input170419.hpp"
//#include "nn/move170405.hpp"
//#include "nn/move170409.hpp"
#include "nn/move170419.hpp"
#ifndef NO_TF
#include "nn/graph.hpp"
#endif
#ifdef LEARN
//#include "nn/datagen.hpp"
#endif

#include "json.h"

#ifdef _WIN32

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

typedef int socklen_t;

#else

#include <unistd.h>
#include <sys/fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <arpa/inet.h>
#include <netdb.h>

#define SOCKET_ERROR 1

typedef int SOCKET;

#endif // _WIN32

void NNServer(){
#ifndef NO_TF
    // Tensorflowのセッション開始と計算グラフ読み込み
    if (psession0 == nullptr) initializeGraph(&psession0, "./pv_graph.pb");
    
    // TCP-IP接続準備
    unsigned short port = 7626;
    
    // サーバー側
    SOCKET mySocket;
    struct sockaddr_in myAddr;
    fd_set fds, readfds;
    
    // クライアント側
    SOCKET csocket;
    struct sockaddr_in caddr;
    
    socklen_t addr_size = sizeof(struct sockaddr_in);
    
    memset(&myAddr, 0, sizeof(struct sockaddr_in));
    memset(&caddr, 0, sizeof(struct sockaddr_in));
    
    struct timeval timeout;
    
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    
#ifdef _WIN32
    // Windows 独自の設定
    WSADATA data;
    if(SOCKET_ERROR == WSAStartup(MAKEWORD(2, 0), &data)){
        cerr << "failed to initialize WSA-data." << std::endl;
        exit(1);
    }
#endif // _WIN32
    
    // ソケットの生成
    mySocket = ::socket(AF_INET, SOCK_STREAM, 0);
    if(mySocket < 0){
        std::cerr << "failed to open server socket." << std::endl;
        exit(1);
    }
    
    // sockaddr_in 構造体のセット
    memset(&myAddr, 0, sizeof(struct sockaddr_in));
    myAddr.sin_port = htons(port);
    myAddr.sin_family = AF_INET;
    myAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    
    int i = 1, j = sizeof(i);
    setsockopt(mySocket, SOL_SOCKET, SO_REUSEADDR, (char *)&i, j);
    
    // ソケットのバインド
    i = bind(mySocket, (struct sockaddr *) &myAddr, sizeof(myAddr));
    
    // 接続の許可
    i = listen(mySocket, 1);
    
    csocket = ::accept(mySocket, (struct sockaddr *) &caddr, &addr_size);
    if(csocket < 0){
        std::cerr << "failed to open client socket." << std::endl;
        exit(1);
    }
    
    // fd_setの初期化
    FD_ZERO(&readfds);
    
    // selectで待つ読み込みソケットとしてmyScoketを登録
    FD_SET(csocket, &readfds);
    
    // 読み込み用fd_setの初期化
    memcpy(&fds, &readfds, sizeof(fd_set));
    
    // fdsに設定されたソケットが読み込み可能になるまで待つ
    select(50, &fds, nullptr, nullptr, &timeout);
    
    dup2(csocket, STDIN_FILENO);
    dup2(csocket, STDOUT_FILENO);
    
    // データ待ち開始
    std::string str;
    while (std::getline(std::cin, str))
    {
        // 受信
        std::vector<std::tuple<uint64_t, std::string, float>> ans;
        std::cerr << ">> " << str << std::endl;
        
        {
            nlohmann::json json = nlohmann::json::parse(str);
            
            auto& request = json["request"];
            
            for(auto& r : request){
                uint64_t key = r[0];
                std::string sfen = r[1];
                ans.push_back(std::make_tuple(key, sfen, 0.0f));
            }
        }
        
        // NN計算
        size_t batchSize = 4;
        std::vector<Position> p(batchSize);
        for (int i = 0; i < (int)ans.size(); i += batchSize)
        {
            int n = std::min(batchSize, ans.size() - i);
            for (int j = 0; j < n; ++j)
                p[j].set(std::get<1>(ans[i + j]), nullptr);
            auto v = getValue(p.data(), n);
            for (int j = 0; j < n; ++j)
                std::get<2>(ans[i + j]) = v[j];
        }
        
        // 送信
        {
            nlohmann::json json;
            json["answer"] = {};
            for (auto& pos : ans)
            {
                json["answer"].push_back({std::get<0>(pos), float(std::get<2>(pos))});
            }
            
            std::string ostr = json.dump();
            std::cerr << "<< " << ostr << std::endl;
            std::cout << ostr << std::endl;
        }
    }
#endif
}

#ifdef LEARN

void csaToHcpr(const std::string& inputPath, const std::string& outputPath, Position& pos) {
    // 棋譜ファイル (CSA) を受け取って各局面を .hcpr 形式で保存する
    std::cerr << "input: " << inputPath << std::endl;
    std::cerr << "output: " << outputPath << std::endl;
    Learner *plearner = new Learner();
    plearner->readBook(pos, inputPath, "-", "-", "-", 0);
    Mutex omutex;
    std::ofstream ofs(outputPath.c_str(), std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot open " << outputPath << std::endl;
        exit(EXIT_FAILURE);
    }
    for(auto& game : plearner->bookMovesDatum_){
        pos.set(DefaultStartPositionSFEN, pos.searcher()->threads.main());
        std::deque<StateInfo> siv;
        for(auto& bm : game){
            const Color myColor = pos.turn();
            const Move move = bm.move;
            HuffmanCodedPosAndResult hcpr;
            hcpr.hcp = pos.toHuffmanCodedPos();
            hcpr.bestMove16 = static_cast<u16>(move.value());
            hcpr.result = bm.winner ? 32600 : -32600;
            std::unique_lock<Mutex> lock(omutex);
            ofs.write(reinterpret_cast<char*>(&hcpr), sizeof(hcpr));
            siv.push_back(StateInfo());
            pos.doMove(move, siv.back());
        }
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}

void usiToHcpr(const std::string& inputPath, const std::string& outputPath, Position& pos) {
    // 棋譜ファイル (USI) を受け取って各局面を .hcpr 形式で保存する
    std::string line;
    Mutex omutex;
    std::ifstream ifs(inputPath);
    if (!ifs) {
        std::cerr << "Error: cannot open " << inputPath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::ofstream ofs(outputPath.c_str(), std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot open " << outputPath << std::endl;
        exit(EXIT_FAILURE);
    }
    int games = 0;
    while (ifs) {
        std::getline(ifs, line);
        std::vector<std::string> usis = split(line, ' ');
        // usi読み取り、試合勝敗を判定 (ルール違反はないと仮定する)
        std::unordered_set<Key> keyHash;
        std::vector<Move> moves;
        std::deque<StateInfo> siv;
        s32 result = -32600; // 勝側が最後の手のはず
        pos.set(DefaultStartPositionSFEN, pos.searcher()->threads.main());
        for (int i = 3; i < (int)usis.size(); ++i) {
            const std::string& usi = usis[i];
            if (usi.size() == 0)continue;
            const Key key = pos.getKey();
            if (keyHash.count(key) < 4) {
                keyHash.insert(key);
            } else { // 千日手
                result = -128;
                break;
            }
            Move move = usiToMove(pos, usi);
            if (!move)break;
            //std::cerr << usi << "->" << move.toUSI() << " ";
            result *= -1;
            siv.push_back(StateInfo());
            pos.doMove(move, siv.back());
            moves.push_back(move);
        }
        // データ保存
        pos.set(DefaultStartPositionSFEN, pos.searcher()->threads.main());
        for (Move move : moves) {
            HuffmanCodedPosAndResult hcpr;
            hcpr.hcp = pos.toHuffmanCodedPos();
            hcpr.bestMove16 = static_cast<u16>(move.value());
            //std::cerr << move.toUSI() << " ";
            hcpr.result = (pos.turn() == Black) ? result : -result;
            //std::cerr << hcpr.bestMove16 << "," << hcpr.result << " ";
            std::unique_lock<Mutex> lock(omutex);
            ofs.write(reinterpret_cast<char*>(&hcpr), sizeof(hcpr));
            siv.push_back(StateInfo());
            pos.doMove(move, siv.back());
        }
        //std::cerr << std::endl;
        ++games;
        if(games % 10000 == 0){
            std::cerr << games << " games." << std::endl;
        }
    }
}

#endif

namespace {
    void onThreads(Searcher* s, const USIOption&)      { s->threads.readUSIOptions(s); }
    void onHashSize(Searcher* s, const USIOption& opt) { s->tt.resize(opt); }
    void onClearHash(Searcher* s, const USIOption&)    { s->tt.clear(); }
}

bool CaseInsensitiveLess::operator () (const std::string& s1, const std::string& s2) const {
    for (size_t i = 0; i < s1.size() && i < s2.size(); ++i) {
        const int c1 = tolower(s1[i]);
        const int c2 = tolower(s2[i]);
        if (c1 != c2)
            return c1 < c2;
    }
    return s1.size() < s2.size();
}

namespace {
    // 論理的なコア数の取得
    inline int cpuCoreCount() {
        // std::thread::hardware_concurrency() は 0 を返す可能性がある。
        // HyperThreading が有効なら論理コア数だけ thread 生成した方が強い。
        return std::max(static_cast<int>(std::thread::hardware_concurrency()), 1);
    }

    class StringToPieceTypeCSA : public std::map<std::string, PieceType> {
    public:
        StringToPieceTypeCSA() {
            (*this)["FU"] = Pawn;
            (*this)["KY"] = Lance;
            (*this)["KE"] = Knight;
            (*this)["GI"] = Silver;
            (*this)["KA"] = Bishop;
            (*this)["HI"] = Rook;
            (*this)["KI"] = Gold;
            (*this)["OU"] = King;
            (*this)["TO"] = ProPawn;
            (*this)["NY"] = ProLance;
            (*this)["NK"] = ProKnight;
            (*this)["NG"] = ProSilver;
            (*this)["UM"] = Horse;
            (*this)["RY"] = Dragon;
        }
        PieceType value(const std::string& str) const {
            return this->find(str)->second;
        }
        bool isLegalString(const std::string& str) const {
            return (this->find(str) != this->end());
        }
    };
    const StringToPieceTypeCSA g_stringToPieceTypeCSA;
}

void OptionsMap::init(Searcher* s) {
    const int MaxHashMB = 1024 * 1024;
    (*this)["USI_Hash"]                    = USIOption(256, 1, MaxHashMB, onHashSize, s);
    (*this)["Clear_Hash"]                  = USIOption(onClearHash, s);
    (*this)["Book_File"]                   = USIOption("book/20150503/book.bin");
    (*this)["Eval_Dir"]                    = USIOption("/Users/ohto/Documents/apery/src/20161007");
    (*this)["Best_Book_Move"]              = USIOption(false);
    (*this)["OwnBook"]                     = USIOption(true);
    (*this)["Min_Book_Ply"]                = USIOption(SHRT_MAX, 0, SHRT_MAX);
    (*this)["Max_Book_Ply"]                = USIOption(SHRT_MAX, 0, SHRT_MAX);
    (*this)["Min_Book_Score"]              = USIOption(-180, -ScoreInfinite, ScoreInfinite);
    (*this)["USI_Ponder"]                  = USIOption(true);
    (*this)["Byoyomi_Margin"]              = USIOption(500, 0, INT_MAX);
    (*this)["Time_Margin"]                 = USIOption(4500, 0, INT_MAX);
    (*this)["MultiPV"]                     = USIOption(1, 1, MaxLegalMoves);
    (*this)["Max_Random_Score_Diff"]       = USIOption(0, 0, ScoreMate0Ply);
    (*this)["Max_Random_Score_Diff_Ply"]   = USIOption(SHRT_MAX, 0, SHRT_MAX);
    (*this)["Slow_Mover_10"]               = USIOption(10, 1, 1000); // 持ち時間15分, 秒読み10秒では10, 持ち時間2時間では3にした。(sdt4)
    (*this)["Slow_Mover_16"]               = USIOption(20, 1, 1000); // 持ち時間15分, 秒読み10秒では50, 持ち時間2時間では20にした。(sdt4)
    (*this)["Slow_Mover_20"]               = USIOption(40, 1, 1000); // 持ち時間15分, 秒読み10秒では50, 持ち時間2時間では40にした。(sdt4)
    (*this)["Slow_Mover"]                  = USIOption(89, 1, 1000);
    (*this)["Draw_Ply"]                    = USIOption(256, 1, INT_MAX);
    (*this)["Move_Overhead"]               = USIOption(30, 0, 5000);
    (*this)["Minimum_Thinking_Time"]       = USIOption(20, 0, INT_MAX);
    (*this)["Threads"]                     = USIOption(cpuCoreCount(), 1, MaxThreads, onThreads, s);
#ifdef NDEBUG
    (*this)["Engine_Name"]                 = USIOption("ShogiNet");
#else
    (*this)["Engine_Name"]                 = USIOption("ShogiNet Debug Build");
#endif
}

USIOption::USIOption(const char* v, Fn* f, Searcher* s) :
    type_("string"), min_(0), max_(0), onChange_(f), searcher_(s)
{
    defaultValue_ = currentValue_ = v;
}

USIOption::USIOption(const bool v, Fn* f, Searcher* s) :
    type_("check"), min_(0), max_(0), onChange_(f), searcher_(s)
{
    defaultValue_ = currentValue_ = (v ? "true" : "false");
}

USIOption::USIOption(Fn* f, Searcher* s) :
    type_("button"), min_(0), max_(0), onChange_(f), searcher_(s) {}

USIOption::USIOption(const int v, const int min, const int max, Fn* f, Searcher* s)
    : type_("spin"), min_(min), max_(max), onChange_(f), searcher_(s)
{
    std::ostringstream ss;
    ss << v;
    defaultValue_ = currentValue_ = ss.str();
}

USIOption& USIOption::operator = (const std::string& v) {
    assert(!type_.empty());

    if ((type_ != "button" && v.empty())
        || (type_ == "check" && v != "true" && v != "false")
        || (type_ == "spin" && (atoi(v.c_str()) < min_ || max_ < atoi(v.c_str()))))
    {
        return *this;
    }

    if (type_ != "button")
        currentValue_ = v;

    if (onChange_ != nullptr)
        (*onChange_)(searcher_, *this);

    return *this;
}

std::ostream& operator << (std::ostream& os, const OptionsMap& om) {
    for (auto& elem : om) {
        const USIOption& o = elem.second;
        os << "\noption name " << elem.first << " type " << o.type_;
        if (o.type_ != "button")
            os << " default " << o.defaultValue_;

        if (o.type_ == "spin")
            os << " min " << o.min_ << " max " << o.max_;
    }
    return os;
}

void go(const Position& pos, std::istringstream& ssCmd) {
    LimitsType limits;
    std::string token;

    limits.startTime.restart();
    
#ifndef NO_TF
    // NN計算
    getBestMove(pos);
    //Position tpos = pos;
    //getBestSearchMove(tpos);
#else
    // 探索
    while (ssCmd >> token) {
        if      (token == "ponder"     ) limits.ponder = true;
        else if (token == "btime"      ) ssCmd >> limits.time[Black];
        else if (token == "wtime"      ) ssCmd >> limits.time[White];
        else if (token == "binc"       ) ssCmd >> limits.inc[Black];
        else if (token == "winc"       ) ssCmd >> limits.inc[White];
        else if (token == "infinite"   ) limits.infinite = true;
        else if (token == "byoyomi" || token == "movetime") ssCmd >> limits.moveTime;
        else if (token == "mate"       ) ssCmd >> limits.mate;
        else if (token == "depth"      ) ssCmd >> limits.depth;
        else if (token == "nodes"      ) ssCmd >> limits.nodes;
        else if (token == "searchmoves") {
            while (ssCmd >> token)
                limits.searchmoves.push_back(usiToMove(pos, token));
        }
    }
    if      (limits.moveTime != 0)
        limits.moveTime -= pos.searcher()->options["Byoyomi_Margin"];
    else if (pos.searcher()->options["Time_Margin"] != 0)
        limits.time[pos.turn()] -= pos.searcher()->options["Time_Margin"];
    pos.searcher()->threads.startThinking(pos, limits, pos.searcher()->states);
#endif
}

#if defined LEARN
// 学習用。通常の go 呼び出しは文字列を扱って高コストなので、大量に探索の開始、終了を行う学習では別の呼び出し方にする。
void go(const Position& pos, const Ply depth, const Move move) {
    LimitsType limits;
    limits.depth = depth;
    limits.searchmoves.push_back(move);
    pos.searcher()->threads.startThinking(pos, limits, pos.searcher()->states);
    pos.searcher()->threads.main()->waitForSearchFinished();
}
void go(const Position& pos, const Ply depth) {
    LimitsType limits;
    limits.depth = depth;
    pos.searcher()->threads.startThinking(pos, limits, pos.searcher()->states);
    pos.searcher()->threads.main()->waitForSearchFinished();
}
#endif

// 評価値 x を勝率にして返す。
// 係数 600 は Ponanza で採用しているらしい値。
inline double sigmoidWinningRate(const double x) {
    return 1.0 / (1.0 + exp(-x/600.0));
}
inline double dsigmoidWinningRate(const double x) {
    const double a = 1.0/600;
    return a * sigmoidWinningRate(x) * (1 - sigmoidWinningRate(x));
}

// 学習でqsearchだけ呼んだ時のPVを取得する為の関数。
// RootMoves が存在しない為、別の関数とする。
template <bool Undo> // 局面を戻し、moves に PV を書き込むなら true。末端の局面に移動したいだけなら false
bool extractPVFromTT(Position& pos, Move* moves, const Move bestMove) {
    StateInfo state[MaxPly+7];
    StateInfo* st = state;
    TTEntry* tte;
    Ply ply = 0;
    Move m;
    bool ttHit;

    tte = pos.csearcher()->tt.probe(pos.getKey(), ttHit);
    if (ttHit && move16toMove(tte->move(), pos) != bestMove)
        return false; // 教師の手と異なる手の場合は学習しないので false。手が無い時は学習するので true
    while (ttHit
           && pos.moveIsPseudoLegal(m = move16toMove(tte->move(), pos))
           && pos.pseudoLegalMoveIsLegal<false, false>(m, pos.pinnedBB())
           && ply < MaxPly
           && (!pos.isDraw(20) || ply < 6))
    {
        if (Undo)
            *moves++ = m;
        pos.doMove(m, *st++);
        ++ply;
        tte = pos.csearcher()->tt.probe(pos.getKey(), ttHit);
    }
    if (Undo) {
        *moves++ = Move::moveNone();
        while (ply)
            pos.undoMove(*(--moves));
    }
    return true;
}

template <bool Undo>
bool qsearch(Position& pos, const u16 bestMove16) {
    //static std::atomic<int> i;
    //StateInfo st;
    Move pv[MaxPly+1];
    Move moves[MaxPly+1];
    SearchStack stack[MaxPly+7];
    SearchStack* ss = stack + 5;
    memset(ss-5, 0, 8 * sizeof(SearchStack));
    (ss-1)->staticEvalRaw.p[0][0] = (ss+0)->staticEvalRaw.p[0][0] = ScoreNotEvaluated;
    ss->pv = pv;
    // 探索の末端がrootと同じ手番に偏るのを防ぐ為に一手進めて探索してみる。
    //if ((i++ & 1) == 0) {
    //  const Move bestMove = move16toMove(Move(bestMove16), pos);
    //  pos.doMove(bestMove, st);
    //}
    if (pos.inCheck())
        pos.searcher()->qsearch<PV, true >(pos, ss, -ScoreInfinite, ScoreInfinite, Depth0);
    else
        pos.searcher()->qsearch<PV, false>(pos, ss, -ScoreInfinite, ScoreInfinite, Depth0);
    const Move bestMove = move16toMove(Move(bestMove16), pos);
    // pv 取得
    return extractPVFromTT<Undo>(pos, moves, bestMove);
}

#if defined USE_GLOBAL
#else
// 教師局面を増やす為、適当に駒を動かす。玉の移動を多めに。王手が掛かっている時は呼ばない事にする。
void randomMove(Position& pos, std::mt19937& mt) {
    StateInfo state[MaxPly+7];
    StateInfo* st = state;
    const Color us = pos.turn();
    const Color them = oppositeColor(us);
    const Square from = pos.kingSquare(us);
    std::uniform_int_distribution<int> dist(0, 1);
    switch (dist(mt)) {
    case 0: { // 玉の25近傍の移動
        ExtMove legalMoves[MaxLegalMoves]; // 玉の移動も含めた普通の合法手
        ExtMove* pms = &legalMoves[0];
        Bitboard kingToBB = pos.bbOf(us).notThisAnd(neighbor5x5Table(from));
        while (kingToBB) {
            const Square to = kingToBB.firstOneFromSQ11();
            const Move move = makeNonPromoteMove<Capture>(King, from, to, pos);
            if (pos.moveIsPseudoLegal<false>(move)
                && pos.pseudoLegalMoveIsLegal<true, false>(move, pos.pinnedBB()))
            {
                (*pms++).move = move;
            }
        }
        if (&legalMoves[0] != pms) { // 手があったなら
            std::uniform_int_distribution<int> moveDist(0, pms - &legalMoves[0] - 1);
            pos.doMove(legalMoves[moveDist(mt)].move, *st++);
            if (dist(mt)) { // 1/2 の確率で相手もランダムに指す事にする。
                MoveList<LegalAll> ml(pos);
                if (ml.size()) {
                    std::uniform_int_distribution<int> moveDist(0, ml.size()-1);
                    pos.doMove((ml.begin() + moveDist(mt))->move, *st++);
                }
            }
        }
        else
            return;
        break;
    }
    case 1: { // 玉も含めた全ての合法手
        bool moved = false;
        for (int i = 0; i < dist(mt) + 1; ++i) { // 自分だけ、または両者ランダムに1手指してみる。
            MoveList<LegalAll> ml(pos);
            if (ml.size()) {
                std::uniform_int_distribution<int> moveDist(0, ml.size()-1);
                pos.doMove((ml.begin() + moveDist(mt))->move, *st++);
                moved = true;
            }
        }
        if (!moved)
            return;
        break;
    }
    default: UNREACHABLE;
    }

    // 違法手が混ざったりするので、一旦 sfen に直して読み込み、過去の手を参照しないようにする。
    std::string sfen = pos.toSFEN();
    std::istringstream ss(sfen);
    setPosition(pos, ss);
}
// 教師局面を作成する。100万局面で34MB。
void make_teacher(std::istringstream& ssCmd) {
    std::string recordFileName;
    std::string outputFileName;
    int threadNum;
    s64 teacherNodes; // 教師局面数
    ssCmd >> recordFileName;
    ssCmd >> outputFileName;
    ssCmd >> threadNum;
    ssCmd >> teacherNodes;
    if (threadNum <= 0) {
        std::cerr << "Error: thread num = " << threadNum << std::endl;
        exit(EXIT_FAILURE);
    }
    if (teacherNodes <= 0) {
        std::cerr << "Error: teacher nodes = " << teacherNodes << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<Searcher> searchers(threadNum);
    std::vector<Position> positions;
    for (auto& s : searchers) {
        s.init();
        const std::string options[] = {"name Threads value 1",
                                       "name MultiPV value 1",
                                       "name USI_Hash value 256",
                                       "name OwnBook value false",
                                       "name Max_Random_Score_Diff value 0"};
        for (auto& str : options) {
            std::istringstream is(str);
            s.setOption(is);
        }
        positions.emplace_back(DefaultStartPositionSFEN, s.threads.main(), s.thisptr);
    }
    std::ifstream ifs(recordFileName.c_str(), std::ifstream::in | std::ifstream::binary | std::ios::ate);
    if (!ifs) {
        std::cerr << "Error: cannot open " << recordFileName << std::endl;
        exit(EXIT_FAILURE);
    }
    const size_t entryNum = ifs.tellg() / sizeof(HuffmanCodedPos);
    std::uniform_int_distribution<s64> inputFileDist(0, entryNum-1);

    Mutex imutex;
    Mutex omutex;
    std::ofstream ofs(outputFileName.c_str(), std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot open " << outputFileName << std::endl;
        exit(EXIT_FAILURE);
    }
    auto func = [&omutex, &ofs, &imutex, &ifs, &inputFileDist, &teacherNodes](Position& pos, std::atomic<s64>& idx, const int threadID) {
        std::mt19937 mt(std::chrono::system_clock::now().time_since_epoch().count() + threadID);
        std::uniform_real_distribution<double> doRandomMoveDist(0.0, 1.0);
        HuffmanCodedPos hcp;
        while (idx < teacherNodes) {
            {
                std::unique_lock<Mutex> lock(imutex);
                ifs.seekg(inputFileDist(mt) * sizeof(HuffmanCodedPos), std::ios_base::beg);
                ifs.read(reinterpret_cast<char*>(&hcp), sizeof(hcp));
            }
            setPosition(pos, hcp);
            randomMove(pos, mt); // 教師局面を増やす為、取得した元局面からランダムに動かしておく。
            double randomMoveRateThresh = 0.2;
            std::unordered_set<Key> keyHash;
            StateListPtr states = StateListPtr(new std::deque<StateInfo>(1));
            for (Ply ply = pos.gamePly(); ply < 400; ++ply, ++idx) { // 400 手くらいで終了しておく。
                if (!pos.inCheck() && doRandomMoveDist(mt) <= randomMoveRateThresh) { // 王手が掛かっていない局面で、randomMoveRateThresh の確率でランダムに局面を動かす。
                    randomMove(pos, mt);
                    ply = 0;
                    randomMoveRateThresh /= 2; // 局面を進めるごとに未知の局面になっていくので、ランダムに動かす確率を半分ずつ減らす。
                }
                const Key key = pos.getKey();
                if (keyHash.find(key) == std::end(keyHash))
                    keyHash.insert(key);
                else // 同一局面 2 回目で千日手判定とする。
                    break;
                pos.searcher()->alpha = -ScoreMaxEvaluate;
                pos.searcher()->beta  =  ScoreMaxEvaluate;
                go(pos, static_cast<Depth>(15));
                const Score score = pos.searcher()->threads.main()->rootMoves[0].score;
                const Move bestMove = pos.searcher()->threads.main()->rootMoves[0].pv[0];
                //if (3000 < abs(score)) // 差が付いたので投了した事にする。
                //    break;
                //else
                if (!bestMove) // 勝ち宣言など
                    break;

                HuffmanCodedPosAndEval hcpe;
                hcpe.hcp = pos.toHuffmanCodedPos();
                auto& pv = pos.searcher()->threads.main()->rootMoves[0].pv;
                hcpe.bestMove16 = static_cast<u16>(pv[0].value());
                const Color rootTurn = pos.turn();
                if (abs(score) > 30000) {
                    hcpe.eval = score;
                } else {
                    StateInfo state[MaxPly+7];
                    StateInfo* st = state;
                    for (size_t i = 0; i < pv.size(); ++i)
                        pos.doMove(pv[i], *st++);
                    // evaluate() の差分計算を無効化する。
                    SearchStack ss[2];
                    ss[0].staticEvalRaw.p[0][0] = ss[1].staticEvalRaw.p[0][0] = ScoreNotEvaluated;
                    const Score eval = evaluate(pos, ss+1);
                    // root の手番から見た評価値に直す。
                    hcpe.eval = (rootTurn == pos.turn() ? eval : -eval);
                    
                    std::cerr << score << " " << hcpe.eval << std::endl;

                    for (size_t i = pv.size(); i > 0;)
                        pos.undoMove(pv[--i]);
                }
                std::unique_lock<Mutex> lock(omutex);
                ofs.write(reinterpret_cast<char*>(&hcpe), sizeof(hcpe));

                states->push_back(StateInfo());
                pos.doMove(bestMove, states->back());
            }
        }
    };
    auto progressFunc = [&teacherNodes] (std::atomic<s64>& index, Timer& t) {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5)); // 指定秒だけ待機し、進捗を表示する。
            const s64 madeTeacherNodes = index;
            const double progress = static_cast<double>(madeTeacherNodes) / teacherNodes;
            auto elapsed_msec = t.elapsed();
            if (progress > 0.0) // 0 除算を回避する。
                std::cout << std::fixed << "Progress: " << std::setprecision(2) << std::min(100.0, progress * 100.0)
                          << "%, Elapsed: " << elapsed_msec/1000
                          << "[s], Remaining: " << std::max<s64>(0, elapsed_msec*(1.0 - progress)/(progress*1000)) << "[s]" << std::endl;
            if (index >= teacherNodes)
                break;
        }
    };
    std::atomic<s64> index;
    index = 0;
    Timer t = Timer::currentTime();
    std::vector<std::thread> threads(threadNum);
    for (int i = 0; i < threadNum; ++i)
        threads[i] = std::thread([&positions, &index, i, &func] { func(positions[i], index, i); });
    std::thread progressThread([&index, &progressFunc, &t] { progressFunc(index, t); });
    for (int i = 0; i < threadNum; ++i)
        threads[i].join();
    progressThread.join();

    std::cout << "Made " << teacherNodes << " teacher nodes in " << t.elapsed()/1000 << " seconds." << std::endl;
}

namespace {
    // Learner とほぼ同じもの。todo: Learner と共通化する。

    using LowerDimensionedEvaluatorGradient = EvaluatorBase<std::array<std::atomic<double>, 2>,
                                                            std::array<std::atomic<double>, 2>,
                                                            std::array<std::atomic<double>, 2> >;
    using EvalBaseType = EvaluatorBase<std::array<double, 2>,
                                       std::array<double, 2>,
                                       std::array<double, 2> >;

    // 小数の評価値を round して整数に直す。
    void copyEval(Evaluator& eval, EvalBaseType& evalBase) {
#if defined _OPENMP
#pragma omp parallel
#endif
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < eval.kpps_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*eval.oneArrayKPP(i))[boardTurn] = round((*evalBase.oneArrayKPP(i))[boardTurn]);
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < eval.kkps_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*eval.oneArrayKKP(i))[boardTurn] = round((*evalBase.oneArrayKKP(i))[boardTurn]);
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < eval.kks_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*eval.oneArrayKK(i))[boardTurn] = round((*evalBase.oneArrayKK(i))[boardTurn]);
    }
    // 整数の評価値を小数に直す。
    void copyEval(EvalBaseType& evalBase, Evaluator& eval) {
#if defined _OPENMP
#pragma omp parallel
#endif
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < evalBase.kpps_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*evalBase.oneArrayKPP(i))[boardTurn] = (*eval.oneArrayKPP(i))[boardTurn];
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < evalBase.kkps_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*evalBase.oneArrayKKP(i))[boardTurn] = (*eval.oneArrayKKP(i))[boardTurn];
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < evalBase.kks_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*evalBase.oneArrayKK(i))[boardTurn] = (*eval.oneArrayKK(i))[boardTurn];
    }
    void averageEval(EvalBaseType& averagedEvalBase, EvalBaseType& evalBase) {
        constexpr double AverageDecay = 0.8; // todo: 過去のデータの重みが強すぎる可能性あり。
#if defined _OPENMP
#pragma omp parallel
#endif
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < averagedEvalBase.kpps_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*averagedEvalBase.oneArrayKPP(i))[boardTurn] = AverageDecay * (*averagedEvalBase.oneArrayKPP(i))[boardTurn] + (1.0 - AverageDecay) * (*evalBase.oneArrayKPP(i))[boardTurn];
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < averagedEvalBase.kkps_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*averagedEvalBase.oneArrayKKP(i))[boardTurn] = AverageDecay * (*averagedEvalBase.oneArrayKKP(i))[boardTurn] + (1.0 - AverageDecay) * (*evalBase.oneArrayKKP(i))[boardTurn];
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < averagedEvalBase.kks_end_index(); ++i)
            for (int boardTurn = 0; boardTurn < 2; ++boardTurn)
                (*averagedEvalBase.oneArrayKK(i))[boardTurn] = AverageDecay * (*averagedEvalBase.oneArrayKK(i))[boardTurn] + (1.0 - AverageDecay) * (*evalBase.oneArrayKK(i))[boardTurn];
    }
    constexpr double FVPenalty() { return (0.001/static_cast<double>(FVScale)); }
    // RMSProp(実質、改造してAdaGradになっている) でパラメータを更新する。
    template <typename T>
    void updateFV(std::array<T, 2>& v, const std::array<std::atomic<double>, 2>& grad, std::array<std::atomic<double>, 2>& msGrad, std::atomic<double>& max) {
        //constexpr double AttenuationRate = 0.99999;
        constexpr double UpdateParam = 100.0; // 更新用のハイパーパラメータ。大きいと不安定になり、小さいと学習が遅くなる。
        constexpr double epsilon = 0.000001; // 0除算防止の定数

        for (int i = 0; i < 2; ++i) {
            // ほぼAdaGrad
            msGrad[i] = /*AttenuationRate * */msGrad[i] + /*(1.0 - AttenuationRate) * */grad[i] * grad[i];
            const double updateStep = UpdateParam * grad[i] / sqrt(msGrad[i] + epsilon);
            v[i] += updateStep;
            const double fabsmax = fabs(updateStep);
            if (max < fabsmax)
                max = fabsmax;
        }
    }
    void updateEval(EvalBaseType& evalBase,
                    LowerDimensionedEvaluatorGradient& lowerDimentionedEvaluatorGradient,
                    LowerDimensionedEvaluatorGradient& meanSquareOfLowerDimensionedEvaluatorGradient)
    {
        std::atomic<double> max;
        max = 0.0;
#if defined _OPENMP
#pragma omp parallel
#endif
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < evalBase.kpps_end_index(); ++i)
            updateFV(*evalBase.oneArrayKPP(i), *lowerDimentionedEvaluatorGradient.oneArrayKPP(i), *meanSquareOfLowerDimensionedEvaluatorGradient.oneArrayKPP(i), max);
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < evalBase.kkps_end_index(); ++i)
            updateFV(*evalBase.oneArrayKKP(i), *lowerDimentionedEvaluatorGradient.oneArrayKKP(i), *meanSquareOfLowerDimensionedEvaluatorGradient.oneArrayKKP(i), max);
#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < evalBase.kks_end_index(); ++i)
            updateFV(*evalBase.oneArrayKK(i), *lowerDimentionedEvaluatorGradient.oneArrayKK(i), *meanSquareOfLowerDimensionedEvaluatorGradient.oneArrayKK(i), max);

        std::cout << "max update step : " << std::fixed << std::setprecision(2) << max << std::endl;
    }
}

constexpr s64 NodesPerIteration = 1000000; // 1回評価値を更新するのに使う教師局面数

void use_teacher(Position& pos, std::istringstream& ssCmd) {
    std::string teacherFileName;
    int threadNum;
    ssCmd >> teacherFileName;
    ssCmd >> threadNum;
    if (threadNum <= 0)
        exit(EXIT_FAILURE);
    std::vector<Searcher> searchers(threadNum);
    std::vector<Position> positions;
    // std::vector<TriangularEvaluatorGradient> だと、非常に大きな要素が要素数分メモリ上に連続する必要があり、
    // 例えメモリ量が余っていても、連続で確保出来ない場合は bad_alloc してしまうので、unordered_map にする。
    std::unordered_map<int, std::unique_ptr<TriangularEvaluatorGradient> > evaluatorGradients;
    // evaluatorGradients(threadNum) みたいにコンストラクタで確保するとスタックを使い切って落ちたので emplace_back する。
    for (int i = 0; i < threadNum; ++i)
        evaluatorGradients.emplace(i, std::move(std::unique_ptr<TriangularEvaluatorGradient>(new TriangularEvaluatorGradient)));
    for (auto& s : searchers) {
        s.init();
        const std::string options[] = {"name Threads value 1",
                                       "name MultiPV value 1",
                                       "name USI_Hash value 256",
                                       "name OwnBook value false",
                                       "name Max_Random_Score_Diff value 0"};
        for (auto& str : options) {
            std::istringstream is(str);
            s.setOption(is);
        }
        positions.emplace_back(DefaultStartPositionSFEN, s.threads.main(), s.thisptr);
    }
    if (teacherFileName == "-") // "-" なら棋譜ファイルを読み込まない。
        exit(EXIT_FAILURE);
    std::ifstream ifs(teacherFileName.c_str(), std::ios::binary);
    if (!ifs)
        exit(EXIT_FAILURE);

    Mutex mutex;
    auto func = [&mutex, &ifs](Position& pos, TriangularEvaluatorGradient& evaluatorGradient, double& loss, std::atomic<s64>& nodes) {
        SearchStack ss[2];
        HuffmanCodedPosAndEval hcpe;
        evaluatorGradient.clear();
        pos.searcher()->tt.clear();
        while (true) {
            {
                std::unique_lock<Mutex> lock(mutex);
                if (NodesPerIteration < nodes++)
                    return;
                ifs.read(reinterpret_cast<char*>(&hcpe), sizeof(hcpe));
                if (ifs.eof())
                    return;
            }
            auto setpos = [](HuffmanCodedPosAndEval& hcpe, Position& pos) {
                setPosition(pos, hcpe.hcp);
            };
            setpos(hcpe, pos);
            const Color rootColor = pos.turn();
            pos.searcher()->alpha = -ScoreMaxEvaluate;
            pos.searcher()->beta  =  ScoreMaxEvaluate;
            if (!qsearch<false>(pos, hcpe.bestMove16)) // 末端の局面に移動する。
                continue;
            // pv を辿って評価値を返す。pos は pv を辿る為に状態が変わる。
            auto pvEval = [&ss, &rootColor](Position& pos) {
                ss[0].staticEvalRaw.p[0][0] = ss[1].staticEvalRaw.p[0][0] = ScoreNotEvaluated;
                // evaluate() は手番側から見た点数なので、eval は rootColor から見た点数。
                const Score eval = (rootColor == pos.turn() ? evaluate(pos, ss+1) : -evaluate(pos, ss+1));
                return eval;
            };
            const Score eval = pvEval(pos);
            const Score teacherEval = static_cast<Score>(hcpe.eval); // root から見た評価値が入っている。
            const Color leafColor = pos.turn(); // pos は末端の局面になっている。
            // x を浅い読みの評価値、y を深い読みの評価値として、
            // 目的関数 f(x, y) は、勝率の誤差の最小化を目指す以下の式とする。
            // また、** 2 は 2 乗を表すとする。
            // f(x,y) = (sigmoidWinningRate(x) - sigmoidWinningRate(y)) ** 2
            //        = sigmoidWinningRate(x)**2 - 2*sigmoidWinningRate(x)*sigmoidWinningRate(y) + sigmoidWinningRate(y)**2
            // 浅い読みの点数を修正したいので、x について微分すると。
            // df(x,y)/dx = 2*sigmoidWinningRate(x)*dsigmoidWinningRate(x)-2*sigmoidWinningRate(y)*dsigmoidWinningRate(x)
            //            = 2*dsigmoidWinningRate(x)*(sigmoidWinningRate(x) - sigmoidWinningRate(y))
            const double dsig = 2*dsigmoidWinningRate(eval)*(sigmoidWinningRate(eval) - sigmoidWinningRate(teacherEval));
            const double tmp = sigmoidWinningRate(eval) - sigmoidWinningRate(teacherEval);
            loss += tmp * tmp;
            std::array<double, 2> dT = {{(rootColor == Black ? -dsig : dsig), (rootColor == leafColor ? -dsig : dsig)}};
            evaluatorGradient.incParam(pos, dT);
        }
    };

    auto lowerDimensionedEvaluatorGradient = std::unique_ptr<LowerDimensionedEvaluatorGradient>(new LowerDimensionedEvaluatorGradient);
    auto meanSquareOfLowerDimensionedEvaluatorGradient = std::unique_ptr<LowerDimensionedEvaluatorGradient>(new LowerDimensionedEvaluatorGradient); // 過去の gradient の mean square (二乗総和)
    auto evalBase = std::unique_ptr<EvalBaseType>(new EvalBaseType); // double で保持した評価関数の要素。相対位置などに分解して保持する。
    auto averagedEvalBase = std::unique_ptr<EvalBaseType>(new EvalBaseType); // ファイル保存する際に評価ベクトルを平均化したもの。
    auto eval = std::unique_ptr<Evaluator>(new Evaluator); // 整数化した評価関数。相対位置などに分解して保持する。
    eval->init(pos.searcher()->options["Eval_Dir"], false);
    copyEval(*evalBase, *eval); // 小数に直してコピー。
    memcpy(averagedEvalBase.get(), evalBase.get(), sizeof(EvalBaseType));
    const size_t fileSize = static_cast<size_t>(ifs.seekg(0, std::ios::end).tellg());
    ifs.clear(); // 読み込み完了をクリアする。
    ifs.seekg(0, std::ios::beg); // ストリームポインタを先頭に戻す。
    const s64 MaxNodes = fileSize / sizeof(HuffmanCodedPosAndEval);
    std::atomic<s64> nodes; // 今回のイテレーションで読み込んだ学習局面数。
    auto writeEval = [&] {
        // ファイル保存
        copyEval(*eval, *averagedEvalBase); // 平均化した物を整数の評価値にコピー
        //copyEval(*eval, *evalBase); // 平均化せずに整数の評価値にコピー
        std::cout << "write eval ... " << std::flush;
        eval->write(pos.searcher()->options["Eval_Dir"]);
        std::cout << "done" << std::endl;
    };
    // 平均化していない合成後の評価関数バイナリも出力しておく。
    auto writeSyn = [&] {
        std::ofstream((Evaluator::addSlashIfNone(pos.searcher()->options["Eval_Dir"]) + "KPP_synthesized.bin").c_str()).write((char*)Evaluator::KPP, sizeof(Evaluator::KPP));
        std::ofstream((Evaluator::addSlashIfNone(pos.searcher()->options["Eval_Dir"]) + "KKP_synthesized.bin").c_str()).write((char*)Evaluator::KKP, sizeof(Evaluator::KKP));
        std::ofstream((Evaluator::addSlashIfNone(pos.searcher()->options["Eval_Dir"]) + "KK_synthesized.bin" ).c_str()).write((char*)Evaluator::KK , sizeof(Evaluator::KK ));
    };
    Timer t;
    // 教師データ全てから学習した時点で終了する。
    for (s64 iteration = 0; NodesPerIteration * iteration + nodes <= MaxNodes; ++iteration) {
        t.restart();
        nodes = 0;
        std::cout << "iteration: " << iteration << ", nodes: " << NodesPerIteration * iteration + nodes << "/" << MaxNodes
                  << " (" << std::fixed << std::setprecision(2) << static_cast<double>(NodesPerIteration * iteration + nodes) * 100 / MaxNodes << "%)" << std::endl;
        std::vector<std::thread> threads(threadNum);
        std::vector<double> losses(threadNum, 0.0);
        for (int i = 0; i < threadNum; ++i)
            threads[i] = std::thread([&positions, i, &func, &evaluatorGradients, &losses, &nodes] { func(positions[i], *(evaluatorGradients[i]), losses[i], nodes); });
        for (int i = 0; i < threadNum; ++i)
            threads[i].join();
        if (nodes < NodesPerIteration)
            break; // パラメータ更新するにはデータが足りなかったので、パラメータ更新せずに終了する。

        for (size_t size = 1; size < (size_t)threadNum; ++size)
            *(evaluatorGradients[0]) += *(evaluatorGradients[size]); // 複数スレッドで個別に保持していた gradients を [0] の要素に集約する。
        lowerDimensionedEvaluatorGradient->clear();
        lowerDimension(*lowerDimensionedEvaluatorGradient, *(evaluatorGradients[0]));

        updateEval(*evalBase, *lowerDimensionedEvaluatorGradient, *meanSquareOfLowerDimensionedEvaluatorGradient);
        averageEval(*averagedEvalBase, *evalBase); // 平均化する。
        if (iteration < 10) // 最初は値の変動が大きいので適当に変動させないでおく。
            memset(&(*evalBase), 0, sizeof(EvalBaseType));
        if (iteration % 100 == 0) {
            writeEval();
            writeSyn();
        }
        copyEval(*eval, *evalBase); // 整数の評価値にコピー
        eval->init(pos.searcher()->options["Eval_Dir"], false, false); // 探索で使う評価関数の更新
        g_evalTable.clear(); // 評価関数のハッシュテーブルも更新しないと、これまで探索した評価値と矛盾が生じる。
        std::cout << "iteration elapsed: " << t.elapsed() / 1000 << "[sec]" << std::endl;
        std::cout << "loss: " << std::accumulate(std::begin(losses), std::end(losses), 0.0) << std::endl;
        printEvalTable(SQ88, f_gold + SQ78, f_gold, false);
    }
    writeEval();
    writeSyn();
}

// 教師データが壊れていないかチェックする。
// todo: 教師データがたまに壊れる原因を調べる。
void check_teacher(std::istringstream& ssCmd) {
    std::string teacherFileName;
    int threadNum;
    ssCmd >> teacherFileName;
    ssCmd >> threadNum;
    if (threadNum <= 0)
        exit(EXIT_FAILURE);
    std::vector<Searcher> searchers(threadNum);
    std::vector<Position> positions;
    for (auto& s : searchers) {
        s.init();
        positions.emplace_back(DefaultStartPositionSFEN, s.threads.main(), s.thisptr);
    }
    std::ifstream ifs(teacherFileName.c_str(), std::ios::binary);
    if (!ifs)
        exit(EXIT_FAILURE);
    Mutex mutex;
    auto func = [&mutex, &ifs](Position& pos) {
        HuffmanCodedPosAndEval hcpe;
        while (true) {
            {
                std::unique_lock<Mutex> lock(mutex);
                ifs.read(reinterpret_cast<char*>(&hcpe), sizeof(hcpe));
                if (ifs.eof())
                    return;
            }
            if (!setPosition(pos, hcpe.hcp))
                exit(EXIT_FAILURE);
        }
    };
    std::vector<std::thread> threads(threadNum);
    for (int i = 0; i < threadNum; ++i)
        threads[i] = std::thread([&positions, i, &func] { func(positions[i]); });
    for (int i = 0; i < threadNum; ++i)
        threads[i].join();
    exit(EXIT_SUCCESS);
}
#endif

Move usiToMoveBody(const Position& pos, const std::string& moveStr) {
    Move move;
    if (g_charToPieceUSI.isLegalChar(moveStr[0])) {
        // drop
        const PieceType ptTo = pieceToPieceType(g_charToPieceUSI.value(moveStr[0]));
        if (moveStr[1] != '*')
            return Move::moveNone();
        const File toFile = charUSIToFile(moveStr[2]);
        const Rank toRank = charUSIToRank(moveStr[3]);
        if (!isInSquare(toFile, toRank))
            return Move::moveNone();
        const Square to = makeSquare(toFile, toRank);
        move = makeDropMove(ptTo, to);
    }
    else {
        const File fromFile = charUSIToFile(moveStr[0]);
        const Rank fromRank = charUSIToRank(moveStr[1]);
        if (!isInSquare(fromFile, fromRank))
            return Move::moveNone();
        const Square from = makeSquare(fromFile, fromRank);
        const File toFile = charUSIToFile(moveStr[2]);
        const Rank toRank = charUSIToRank(moveStr[3]);
        if (!isInSquare(toFile, toRank))
            return Move::moveNone();
        const Square to = makeSquare(toFile, toRank);
        if (moveStr[4] == '\0')
            move = makeNonPromoteMove<Capture>(pieceToPieceType(pos.piece(from)), from, to, pos);
        else if (moveStr[4] == '+') {
            if (moveStr[5] != '\0')
                return Move::moveNone();
            move = makePromoteMove<Capture>(pieceToPieceType(pos.piece(from)), from, to, pos);
        }
        else
            return Move::moveNone();
    }

    if (pos.moveIsPseudoLegal<false>(move)
        && pos.pseudoLegalMoveIsLegal<false, false>(move, pos.pinnedBB()))
    {
        return move;
    }
    return Move::moveNone();
}
#if !defined NDEBUG
// for debug
Move usiToMoveDebug(const Position& pos, const std::string& moveStr) {
    for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
        if (moveStr == ml.move().toUSI())
            return ml.move();
    }
    return Move::moveNone();
}
Move csaToMoveDebug(const Position& pos, const std::string& moveStr) {
    for (MoveList<LegalAll> ml(pos); !ml.end(); ++ml) {
        if (moveStr == ml.move().toCSA())
            return ml.move();
    }
    return Move::moveNone();
}
#endif
Move usiToMove(const Position& pos, const std::string& moveStr) {
    const Move move = usiToMoveBody(pos, moveStr);
    assert(move == usiToMoveDebug(pos, moveStr));
    return move;
}

Move csaToMoveBody(const Position& pos, const std::string& moveStr) {
    if (moveStr.size() != 6)
        return Move::moveNone();
    const File toFile = charCSAToFile(moveStr[2]);
    const Rank toRank = charCSAToRank(moveStr[3]);
    if (!isInSquare(toFile, toRank))
        return Move::moveNone();
    const Square to = makeSquare(toFile, toRank);
    const std::string ptToString(moveStr.begin() + 4, moveStr.end());
    if (!g_stringToPieceTypeCSA.isLegalString(ptToString))
        return Move::moveNone();
    const PieceType ptTo = g_stringToPieceTypeCSA.value(ptToString);
    Move move;
    if (moveStr[0] == '0' && moveStr[1] == '0')
        // drop
        move = makeDropMove(ptTo, to);
    else {
        const File fromFile = charCSAToFile(moveStr[0]);
        const Rank fromRank = charCSAToRank(moveStr[1]);
        if (!isInSquare(fromFile, fromRank))
            return Move::moveNone();
        const Square from = makeSquare(fromFile, fromRank);
        PieceType ptFrom = pieceToPieceType(pos.piece(from));
        if (ptFrom == ptTo)
            // non promote
            move = makeNonPromoteMove<Capture>(ptFrom, from, to, pos);
        else if (ptFrom + PTPromote == ptTo)
            // promote
            move = makePromoteMove<Capture>(ptFrom, from, to, pos);
        else
            return Move::moveNone();
    }

    if (pos.moveIsPseudoLegal<false>(move)
        && pos.pseudoLegalMoveIsLegal<false, false>(move, pos.pinnedBB()))
    {
        return move;
    }
    return Move::moveNone();
}
Move csaToMove(const Position& pos, const std::string& moveStr) {
    const Move move = csaToMoveBody(pos, moveStr);
    assert(move == csaToMoveDebug(pos, moveStr));
    return move;
}

void setPosition(Position& pos, std::istringstream& ssCmd) {
    std::string token;
    std::string sfen;

    ssCmd >> token;

    if (token == "startpos") {
        sfen = DefaultStartPositionSFEN;
        ssCmd >> token; // "moves" が入力されるはず。
    }
    else if (token == "sfen") {
        while (ssCmd >> token && token != "moves")
            sfen += token + " ";
    }
    else
        return;

    pos.set(sfen, pos.searcher()->threads.main());
    pos.searcher()->states = StateListPtr(new std::deque<StateInfo>(1));

    Ply currentPly = pos.gamePly();
    while (ssCmd >> token) {
        const Move move = usiToMove(pos, token);
        if (!move) break;
        pos.searcher()->states->push_back(StateInfo());
        pos.doMove(move, pos.searcher()->states->back());
        ++currentPly;
    }
    pos.setStartPosPly(currentPly);
}

bool setPosition(Position& pos, const HuffmanCodedPos& hcp) {
    return pos.set(hcp, pos.searcher()->threads.main());
}

void Searcher::setOption(std::istringstream& ssCmd) {
    std::string token;
    std::string name;
    std::string value;

    ssCmd >> token; // "name" が入力されるはず。

    ssCmd >> name;
    // " " が含まれた名前も扱う。
    while (ssCmd >> token && token != "value")
        name += " " + token;

    ssCmd >> value;
    // " " が含まれた値も扱う。
    while (ssCmd >> token)
        value += " " + token;

    if (!options.isLegalOption(name))
        std::cout << "No such option: " << name << std::endl;
    else
        options[name] = value;
}

#if !defined MINIMUL
// for debug
// 指し手生成の速度を計測
void measureGenerateMoves(const Position& pos) {
    pos.print();

    ExtMove legalMoves[MaxLegalMoves];
    for (int i = 0; i < MaxLegalMoves; ++i) legalMoves[i].move = moveNone();
    ExtMove* pms = &legalMoves[0];
    const u64 num = 5000000;
    Timer t = Timer::currentTime();
    if (pos.inCheck()) {
        for (u64 i = 0; i < num; ++i) {
            pms = &legalMoves[0];
            pms = generateMoves<Evasion>(pms, pos);
        }
    }
    else {
        for (u64 i = 0; i < num; ++i) {
            pms = &legalMoves[0];
            pms = generateMoves<CapturePlusPro>(pms, pos);
            pms = generateMoves<NonCaptureMinusPro>(pms, pos);
            pms = generateMoves<Drop>(pms, pos);
//          pms = generateMoves<PseudoLegal>(pms, pos);
//          pms = generateMoves<Legal>(pms, pos);
        }
    }
    const int elapsed = t.elapsed();
    std::cout << "elapsed = " << elapsed << " [msec]" << std::endl;
    if (elapsed != 0)
        std::cout << "times/s = " << num * 1000 / elapsed << " [times/sec]" << std::endl;
    const ptrdiff_t count = pms - &legalMoves[0];
    std::cout << "num of moves = " << count << std::endl;
    for (int i = 0; i < count; ++i)
        std::cout << legalMoves[i].move.toCSA() << ", ";
    std::cout << std::endl;
}
#endif

void Searcher::doUSICommandLoop(int argc, char* argv[]) {
    bool evalTableIsRead = false;
    Position pos(DefaultStartPositionSFEN, threads.main(), thisptr);

    std::string cmd;
    std::string token;

    for (int i = 1; i < argc; ++i)
        cmd += std::string(argv[i]) + " ";

    do {
        if (argc == 1 && !std::getline(std::cin, cmd))
            cmd = "quit";

        std::istringstream ssCmd(cmd);

        ssCmd >> std::skipws >> token;

        if (token == "quit" || token == "stop" || token == "ponderhit" || token == "gameover") {
            if (token != "ponderhit" || signals.stopOnPonderHit) {
                signals.stop = true;
                threads.main()->startSearching(true);
            }
            else
                limits.ponder = false;
            if (token == "ponderhit" && limits.moveTime != 0)
                limits.moveTime += timeManager.elapsed();
        }
        else if (token == "go"       ) go(pos, ssCmd);
        else if (token == "position" ) setPosition(pos, ssCmd);
        else if (token == "usinewgame"); // isready で準備は出来たので、対局開始時に特にする事はない。
        else if (token == "usi"      ) SYNCCOUT << "id name " << std::string(options["Engine_Name"])
                                                << "\nid author Hiraoka Takuya"
                                                << "\n" << options
                                                << "\nusiok" << SYNCENDL;
        else if (token == "isready"  ) { // 対局開始前の準備。
            tt.clear();
            threads.main()->previousScore = ScoreInfinite;
            if (!evalTableIsRead) {
                // 一時オブジェクトを生成して Evaluator::init() を呼んだ直後にオブジェクトを破棄する。
                // 評価関数の次元下げをしたデータを格納する分のメモリが無駄な為、
                std::unique_ptr<Evaluator>(new Evaluator)->init(options["Eval_Dir"], true);
                evalTableIsRead = true;
            }
#ifndef NO_TF
            // Tensorflowのセッション開始と計算グラフ読み込み
            if (psession0 == nullptr) initializeGraph(&psession0, "./pv_graph.pb");
#endif
            SYNCCOUT << "readyok" << SYNCENDL;
        }
        else if (token == "setoption") setOption(ssCmd);
        else if (token == "write_eval") { // 対局で使う為の評価関数バイナリをファイルに書き出す。
            if (!evalTableIsRead)
                std::unique_ptr<Evaluator>(new Evaluator)->init(options["Eval_Dir"], true);
            Evaluator::writeSynthesized(options["Eval_Dir"]);
        }
        else if (token == "nnserv") { // NNサーバー
            NNServer();
        }
#if defined LEARN
        else if (token == "l"        ) {
            auto learner = std::unique_ptr<Learner>(new Learner);
            learner->learn(pos, ssCmd);
        }
        else if (token == "make_teacher") {
            if (!evalTableIsRead) {
                std::unique_ptr<Evaluator>(new Evaluator)->init(options["Eval_Dir"], true);
                evalTableIsRead = true;
            }
            make_teacher(ssCmd);
        }
        else if (token == "use_teacher") {
            if (!evalTableIsRead) {
                std::unique_ptr<Evaluator>(new Evaluator)->init(options["Eval_Dir"], true);
                evalTableIsRead = true;
            }
            use_teacher(pos, ssCmd);
        }
        else if (token == "check_teacher") {
            check_teacher(ssCmd);
        }
        else if (token == "csa2hcpr") {
            std::string inputPath, outputPath;
            ssCmd >> inputPath >> outputPath;
            csaToHcpr(inputPath, outputPath, pos);
        }
        else if (token == "usi2hcpr") {
            std::string inputPath, outputPath;
            ssCmd >> inputPath >> outputPath;
            usiToHcpr(inputPath, outputPath, pos);
        }
        else if (token == "print"    ) printEvalTable(SQ88, f_gold + SQ78, f_gold, false);
#endif
#if !defined MINIMUL
        // 以下、デバッグ用
        else if (token == "bench"    ) {
            if (!evalTableIsRead) {
                std::unique_ptr<Evaluator>(new Evaluator)->init(options["Eval_Dir"], true);
                evalTableIsRead = true;
            }
            benchmark(pos);
        }
        else if (token == "key"      ) SYNCCOUT << pos.getKey() << SYNCENDL;
        else if (token == "tosfen"   ) SYNCCOUT << pos.toSFEN() << SYNCENDL;
        else if (token == "eval"     ) std::cout << evaluateUnUseDiff(pos) / FVScale << std::endl;
        else if (token == "d"        ) pos.print();
        else if (token == "s"        ) measureGenerateMoves(pos);
        else if (token == "t"        ) std::cout << pos.mateMoveIn1Ply().toCSA() << std::endl;
        else if (token == "b"        ) makeBook(pos, ssCmd);
#endif
        else                           SYNCCOUT << "unknown command: " << cmd << SYNCENDL;
    } while (token != "quit" && argc == 1);

    threads.main()->waitForSearchFinished();
}

#ifdef LEARN
int mptd_main(Searcher *const psearcher, int argc, char *argv[]){
    
    std::string csaFilePath = "./2chkifu.csa", outputDir = "./";
    int threads = 1;
    bool testMode = false;
    for(int c = 1; c < argc; ++c){
        if(!strcmp(argv[c], "-l")){
            csaFilePath = std::string(argv[c + 1]);
        }else if(!strcmp(argv[c], "-o")){
            outputDir = std::string(argv[c + 1]);
        }else if(!strcmp(argv[c], "-th")){
            threads = atoi(argv[c + 1]);
        }else if(!strcmp(argv[c], "-ac")){
            testMode = true;
        }
    }
    
    // 出力を逐次確認したい場合のみ
    /*if (psession == nullptr){
        // Tensorflowのセッション開始と計算グラフ読み込み
        initializeGraph("./policy_graph170409.pb");
    }*/
    
    if(testMode){
        calcAccuracy(psearcher, csaFilePath);
    }else{
        //genPolicyTeacher(psearcher, csaFilePath, outputDir, threads);
    }
    
    return 0;
}
#endif

#ifdef PYBIND11_PACKAGE

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::mt19937 mt;

struct PackageInitializer{
    PackageInitializer(){
        initTable();
        Position::initZobrist();
        HuffmanCodedPos::init();
        std::random_device seed;
        mt.seed(seed() ^ (unsigned int)time(NULL));
    }
};

//PackageInitializer _packageInitializer;

void openNNServer(){
    // ニューラルネットのサーバーを立てる
    std::string host = "127.0.0.1";
    unsigned short port = 7626;
    
    
}

py::array_t<float>
getInputsFromSfen(const std::vector<std::string>& sfens){
    const int batchSize = sfens.size();
    const std::vector<int> inputShape = {batchSize, ImageFileNum, ImageRankNum, ImageInputPlains};
    py::array_t<float> inputs(inputShape);
	for(int i = 0; i < batchSize; ++i){
		const std::string& sfen = sfens[i];
		Position pos;
		pos.set(sfen, nullptr);
		const std::vector<int> inputShape = {batchSize, ImageFileNum, ImageRankNum, ImageInputPlains};
		BoardImage image;
		const Color myColor = pos.turn();
        positionToImage(pos, myColor, image); // inputデータ作成
        imageToInput(image, inputs.mutable_data() + ImageFileNum * ImageRankNum * ImageInputPlains * i);
    }
    return inputs;
}

std::tuple<py::array_t<float>, py::array_t<s64>, py::array_t<float>>
getInputsMovesValues(const std::string& teacherFileName, const int batchSize){
    PackageInitializer _packageInitializer;
    // (局面, 着手, 評価値)が記録されたApery形式から受け取る
    const std::vector<int> inputShape = {batchSize, ImageFileNum, ImageRankNum, ImageInputPlains};
    const std::vector<int> moveShape = {batchSize, ImageSupervisedOutputs};
    const std::vector<int> valueShape = {batchSize};
    py::array_t<float> inputs(inputShape);
    py::array_t<s64> moves(moveShape);
    py::array_t<float> values(valueShape);
    
    std::ifstream ifs(teacherFileName.c_str(), std::ios::binary);
    if (!ifs)
        exit(EXIT_FAILURE);
    const std::size_t fileSize = static_cast<std::size_t>(ifs.seekg(0, std::ios::end).tellg());
    const s64 numPositions = fileSize / sizeof(HuffmanCodedPosAndEval); // 総局面数
    if (!fileSize)
        exit(EXIT_FAILURE);
    ifs.seekg(0, std::ios_base::beg);
    for (int i = 0; i < batchSize; ++i) {
        HuffmanCodedPosAndEval hcpe;
        const int index = mt() % numPositions;
        ifs.seekg(index * sizeof(HuffmanCodedPosAndEval), std::ios_base::beg);
        ifs.read(reinterpret_cast<char*>(&hcpe), sizeof(hcpe));
        // 学習データに変換
        Position pos;
        const Move move16 = static_cast<Move>(hcpe.bestMove16);
        const Score eval = static_cast<Score>(hcpe.eval);
        if(!pos.set(hcpe.hcp, nullptr)){
            i -= 1;
            std::cerr << move16.toUSI() << " " << hcpe.eval << "(" << index << " / " << numPositions << ")" << std::endl;
        }
        const Color myColor = pos.turn();
        const Move move = move16toMove(move16, pos);
        
        BoardImage image;
        positionToImage(pos, myColor, image); // inputデータ作成
        moveToFromTo(move, myColor, &image.from, &image.to); // outputデータ作成
        
        imageToInput(image, inputs.mutable_data() + ImageFileNum * ImageRankNum * ImageInputPlains * i);
        imageToMove(image.from, image.to, moves.mutable_data() + ImageSupervisedOutputs * i);
        values.mutable_data()[i] = 2.0 / (1.0 + exp(-double(eval) / 600.0)) - 1.0;
    }
    
    return std::make_tuple(inputs, moves, values);
}

/*auto gen_inputs_moves_values_results(const std::string& fileName, const int num){
 // 勝敗情報も記録されたelmo形式から受け取る
 py::array_t<double>
 }*/

/*std::tuple<py::array_t<float>, py::array_t<s64>, py::array_t<float>>
getMoveValuesBySearch(const std::string& teacherFileName, const int batchSize){
    // 探索により評価した局面の各候補手の評価値を返す
    // データ生成に使用する局面ファイル名を渡す
    std::random_device seed;
    std::mt19937 mt(seed() ^ (unsigned int)time(NULL));
    initTable();
    Position::initZobrist();
    HuffmanCodedPos::init();
    
    const std::vector<int> inputShape = {batchSize, ImageFileNum, ImageRankNum, ImageInputPlains};
    const std::vector<int> moveValueShape = {batchSize, move_max, ImageSupervisedOutputs + 1};
    py::array_t<float> inputs(inputShape);
    py::array_t<> moves(moveValueShape);
    
    std::ifstream ifs(teacherFileName.c_str(), std::ios::binary);
    if (!ifs)
        exit(EXIT_FAILURE);
    const std::size_t fileSize = static_cast<std::size_t>(ifs.seekg(0, std::ios::end).tellg());
    const s64 numPositions = fileSize / sizeof(HuffmanCodedPosAndEval); // 総局面数
    if (!fileSize)
        exit(EXIT_FAILURE);
    ifs.seekg(0, std::ios_base::beg);
    
    for (int i = 0; i < batchSize; ++i) {
        HuffmanCodedPosAndEval hcpe;
        const int index = mt() % numPositions;
        ifs.seekg(index * sizeof(HuffmanCodedPosAndEval), std::ios_base::beg);
        ifs.read(reinterpret_cast<char*>(&hcpe), sizeof(hcpe));
        // 学習データに変換
        Position pos;
        pos.set(hcpe.hcp, nullptr);
        const Color myColor = pos.turn();
        const Move move = move16toMove(static_cast<Move>(hcpe.bestMove16), pos);
        // 探索して各合法手の評価値を計算
        go(pos, static_cast<Depth>(15));
        const Move bestMove = pos.searcher()->threads.main()->rootMoves[i].pv[0];
        if (!bestMove) { // 勝ち宣言など
            i -= 1; // データが空かないように
            continue;
        }
        for (int j = 0; j < pos.searcher()->threads.main()->rootMoves; ++j) {
            const RootMove& rm = pos.searcher()->threads.main()->rootMoves[j];
            const Score score = rm.score;
            const Move move = rm.pv[0];
            moveValues[i * move_max  j] = 1;
        }

        
        BoardImage image;
        positionToImage(pos, myColor, image); // inputデータ作成
        moveToFromTo(move, myColor, &image.from, &image.to); // outputデータ作成
        
        imageToInput(image, inputs.mutable_data() + ImageFileNum * ImageRankNum * ImageInputPlains * i);
        imageToMove(image.from, image.to, moves.mutable_data() + ImageSupervisedOutputs);
        values.mutable_data()[i] = 2.0 / (1.0 + exp(-double(eval) / 600.0)) - 1.0;
    }
    
    return std::make_tuple(inputs, moveValues);
}*/


PYBIND11_PLUGIN(nndata) {
    py::module m("nndata", "data ganerator for neural network");
    m.def("get_inputs_from_sfen", &getInputsFromSfen,
          "A function which returns inputs by sfen positions");
    m.def("get_inputs_moves_values", &getInputsMovesValues,
          "A function which returns inputs, moves and values");
    //m.def("gen_inputs_moves_values_results", &gen_inputs_moves_values_results,
    //      "A function which returns inputs, moves, values and results");
    return m.ptr();
}

#endif
