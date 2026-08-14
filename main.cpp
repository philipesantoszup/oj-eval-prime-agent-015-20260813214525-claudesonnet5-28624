
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

// ---------------- Disk B+ Tree ----------------
// Composite key = (idx string up to 64 bytes, int32 value)
// Only insertion causes node splits; deletion simply removes the key from
// the leaf without rebalancing (allowed because we never need to reclaim
// space and the tree stays valid / reasonably shallow for n <= 100000).

static const int BLOCK_SIZE = 4096;
static const int M = 56;          // max keys per node on disk
static const int IDX_LEN = 64;    // bytes reserved for index string

struct Key {
    char idx[IDX_LEN];
    int32_t val;
};

static inline int keycmp(const Key &a, const Key &b) {
    int c = memcmp(a.idx, b.idx, IDX_LEN);
    if (c != 0) return c;
    if (a.val < b.val) return -1;
    if (a.val > b.val) return 1;
    return 0;
}

#pragma pack(push, 1)
struct DiskNode {
    int32_t is_leaf;
    int32_t num_keys;
    int32_t next_leaf; // used only for leaf nodes, -1 otherwise
    Key keys[M];
    int32_t children[M + 1];
    char _pad[BLOCK_SIZE - (4*3 + sizeof(Key)*M + 4*(M+1))];
};
#pragma pack(pop)

struct DiskHeader {
    int32_t root_block;
    int32_t num_blocks; // total blocks including header (block 0)
    char _pad[BLOCK_SIZE - 8];
};

static FILE *fp = nullptr;

static void readBlock(int32_t block_id, void *buf) {
    fseek(fp, (long long)block_id * BLOCK_SIZE, SEEK_SET);
    size_t r = fread(buf, 1, BLOCK_SIZE, fp);
    (void)r;
}

static void writeBlock(int32_t block_id, const void *buf) {
    fseek(fp, (long long)block_id * BLOCK_SIZE, SEEK_SET);
    fwrite(buf, 1, BLOCK_SIZE, fp);
}

static DiskHeader header;

static void readHeader() {
    readBlock(0, &header);
}
static void writeHeader() {
    writeBlock(0, &header);
}

static int32_t allocateBlock() {
    int32_t id = header.num_blocks;
    header.num_blocks++;
    return id;
}

static void initStorage(const char *fname) {
    fp = fopen(fname, "r+b");
    if (fp == nullptr) {
        // create new file
        fp = fopen(fname, "w+b");
        header.root_block = 1;
        header.num_blocks = 2;
        writeHeader();
        DiskNode root{};
        root.is_leaf = 1;
        root.num_keys = 0;
        root.next_leaf = -1;
        writeBlock(1, &root);
    } else {
        readHeader();
    }
}

static void closeStorage() {
    if (fp) { fflush(fp); fclose(fp); fp = nullptr; }
}

// staging node used during insertion (allows temporary overflow of 1 extra key)
struct StageNode {
    int32_t is_leaf;
    int32_t num_keys;
    int32_t next_leaf;
    Key keys[M + 1];
    int32_t children[M + 2];
};

static void toStage(const DiskNode &d, StageNode &s) {
    s.is_leaf = d.is_leaf;
    s.num_keys = d.num_keys;
    s.next_leaf = d.next_leaf;
    for (int i = 0; i < d.num_keys; i++) s.keys[i] = d.keys[i];
    if (!d.is_leaf) {
        for (int i = 0; i <= d.num_keys; i++) s.children[i] = d.children[i];
    }
}

static void toDisk(const StageNode &s, DiskNode &d) {
    d.is_leaf = s.is_leaf;
    d.num_keys = s.num_keys;
    d.next_leaf = s.next_leaf;
    for (int i = 0; i < s.num_keys; i++) d.keys[i] = s.keys[i];
    if (!s.is_leaf) {
        for (int i = 0; i <= s.num_keys; i++) d.children[i] = s.children[i];
    }
}

// find first index i in [0,num_keys) such that target < keys[i]  (upper bound by strict less)
static int upperBoundPos(const Key *keys, int num_keys, const Key &target) {
    int lo = 0, hi = num_keys;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (keycmp(target, keys[mid]) < 0) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// find first index i in [0,num_keys) such that keys[i] >= target (lower bound)
static int lowerBoundPos(const Key *keys, int num_keys, const Key &target) {
    int lo = 0, hi = num_keys;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (keycmp(keys[mid], target) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static void makeKey(Key &k, const std::string &idxStr, int32_t val) {
    memset(k.idx, 0, IDX_LEN);
    size_t len = idxStr.size();
    if (len > IDX_LEN) len = IDX_LEN;
    memcpy(k.idx, idxStr.data(), len);
    k.val = val;
}

// ---------- INSERT ----------
static const int MAX_DEPTH = 64;

static void insertKey(const Key &newKey) {
    int32_t pathBlocks[MAX_DEPTH];
    int depth = 0;
    int32_t cur = header.root_block;
    DiskNode dnode;

    // descend to leaf, recording path
    while (true) {
        readBlock(cur, &dnode);
        pathBlocks[depth++] = cur;
        if (dnode.is_leaf) break;
        int pos = upperBoundPos(dnode.keys, dnode.num_keys, newKey);
        cur = dnode.children[pos];
    }

    // process leaf
    StageNode stage;
    toStage(dnode, stage);
    int pos = lowerBoundPos(stage.keys, stage.num_keys, newKey);
    if (pos < stage.num_keys && keycmp(stage.keys[pos], newKey) == 0) {
        // duplicate, ignore (should not happen per problem guarantee)
        return;
    }
    // insert into stage.keys at pos
    for (int i = stage.num_keys; i > pos; i--) stage.keys[i] = stage.keys[i-1];
    stage.keys[pos] = newKey;
    stage.num_keys++;

    int32_t promoteChildRight = -1; // block id of right node created by split (child) to add to parent
    Key promoteKey;
    bool needSplit = stage.num_keys > M;
    bool haveSplit = false;

    int32_t curBlockId = pathBlocks[depth - 1];

    if (!needSplit) {
        DiskNode out;
        toDisk(stage, out);
        writeBlock(curBlockId, &out);
    } else {
        // split leaf
        int total = stage.num_keys;
        int leftCount = (total + 1) / 2;
        int rightCount = total - leftCount;
        StageNode leftS{}, rightS{};
        leftS.is_leaf = 1; rightS.is_leaf = 1;
        leftS.num_keys = leftCount; rightS.num_keys = rightCount;
        for (int i = 0; i < leftCount; i++) leftS.keys[i] = stage.keys[i];
        for (int i = 0; i < rightCount; i++) rightS.keys[i] = stage.keys[leftCount + i];
        int32_t rightBlockId = allocateBlock();
        rightS.next_leaf = stage.next_leaf;
        leftS.next_leaf = rightBlockId;

        DiskNode leftD, rightD;
        toDisk(leftS, leftD);
        toDisk(rightS, rightD);
        writeBlock(curBlockId, &leftD);
        writeBlock(rightBlockId, &rightD);

        promoteKey = rightS.keys[0];
        promoteChildRight = rightBlockId;
        haveSplit = true;
    }

    // propagate splits up through internal nodes
    for (int level = depth - 2; level >= 0 && haveSplit; level--) {
        int32_t blockId = pathBlocks[level];
        DiskNode inode;
        readBlock(blockId, &inode);
        StageNode istage;
        toStage(inode, istage);
        // find position of child that we came from: it's the child whose subtree is pathBlocks[level+1]
        // we need position pos2 such that istage.children[pos2] == pathBlocks[level+1]
        int pos2 = -1;
        for (int i = 0; i <= istage.num_keys; i++) {
            if (istage.children[i] == pathBlocks[level+1]) { pos2 = i; break; }
        }
        // insert promoteKey at position pos2, new child promoteChildRight at pos2+1
        for (int i = istage.num_keys; i > pos2; i--) istage.keys[i] = istage.keys[i-1];
        istage.keys[pos2] = promoteKey;
        for (int i = istage.num_keys + 1; i > pos2 + 1; i--) istage.children[i] = istage.children[i-1];
        istage.children[pos2 + 1] = promoteChildRight;
        istage.num_keys++;

        if (istage.num_keys <= M) {
            DiskNode out;
            toDisk(istage, out);
            writeBlock(blockId, &out);
            haveSplit = false;
        } else {
            // split internal node
            int total = istage.num_keys;
            int mid = total / 2;
            StageNode leftS{}, rightS{};
            leftS.is_leaf = 0; rightS.is_leaf = 0;
            leftS.num_keys = mid;
            rightS.num_keys = total - mid - 1;
            for (int i = 0; i < mid; i++) leftS.keys[i] = istage.keys[i];
            for (int i = 0; i <= mid; i++) leftS.children[i] = istage.children[i];
            for (int i = 0; i < rightS.num_keys; i++) rightS.keys[i] = istage.keys[mid + 1 + i];
            for (int i = 0; i <= rightS.num_keys; i++) rightS.children[i] = istage.children[mid + 1 + i];

            int32_t rightBlockId = allocateBlock();
            DiskNode leftD, rightD;
            toDisk(leftS, leftD);
            toDisk(rightS, rightD);
            writeBlock(blockId, &leftD);
            writeBlock(rightBlockId, &rightD);

            promoteKey = istage.keys[mid];
            promoteChildRight = rightBlockId;
            haveSplit = true;
            // continue loop to propagate further up
        }
    }

    if (haveSplit) {
        // root got split -> create new root
        int32_t newRoot = allocateBlock();
        StageNode rootS{};
        rootS.is_leaf = 0;
        rootS.num_keys = 1;
        rootS.keys[0] = promoteKey;
        rootS.children[0] = header.root_block;
        rootS.children[1] = promoteChildRight;
        DiskNode rootD;
        toDisk(rootS, rootD);
        writeBlock(newRoot, &rootD);
        header.root_block = newRoot;
    }
    writeHeader();
}

// ---------- DELETE ----------
static void deleteKey(const Key &target) {
    int32_t cur = header.root_block;
    DiskNode dnode;
    while (true) {
        readBlock(cur, &dnode);
        if (dnode.is_leaf) break;
        int pos = upperBoundPos(dnode.keys, dnode.num_keys, target);
        cur = dnode.children[pos];
    }
    int pos = lowerBoundPos(dnode.keys, dnode.num_keys, target);
    if (pos < dnode.num_keys && keycmp(dnode.keys[pos], target) == 0) {
        for (int i = pos; i < dnode.num_keys - 1; i++) dnode.keys[i] = dnode.keys[i+1];
        dnode.num_keys--;
        writeBlock(cur, &dnode);
    }
    // not found -> no-op
}

// ---------- FIND ----------
static void findIdx(const std::string &idxStr, std::string &outLine) {
    Key target;
    makeKey(target, idxStr, INT32_MIN);

    int32_t cur = header.root_block;
    DiskNode dnode;
    while (true) {
        readBlock(cur, &dnode);
        if (dnode.is_leaf) break;
        int pos = upperBoundPos(dnode.keys, dnode.num_keys, target);
        cur = dnode.children[pos];
    }
    int pos = lowerBoundPos(dnode.keys, dnode.num_keys, target);

    char idxbuf[IDX_LEN + 1];
    memset(idxbuf, 0, sizeof(idxbuf));
    size_t len = idxStr.size();
    if (len > IDX_LEN) len = IDX_LEN;
    memcpy(idxbuf, idxStr.data(), len);

    outLine.clear();
    bool any = false;
    char numbuf[16];
    while (true) {
        while (pos < dnode.num_keys) {
            if (memcmp(dnode.keys[pos].idx, idxbuf, IDX_LEN) != 0) {
                goto done;
            }
            if (any) outLine.push_back(' ');
            int n = snprintf(numbuf, sizeof(numbuf), "%d", dnode.keys[pos].val);
            outLine.append(numbuf, n);
            any = true;
            pos++;
        }
        if (dnode.next_leaf == -1) break;
        int32_t nb = dnode.next_leaf;
        readBlock(nb, &dnode);
        pos = 0;
    }
done:
    if (!any) outLine = "null";
}

int main() {
    initStorage("fs_data.bin");

    int n;
    if (scanf("%d", &n) != 1) { closeStorage(); return 0; }

    std::string cmd, idxStr;
    long long valLL;
    std::string outBuf;
    outBuf.reserve(1 << 16);

    char cmdbuf[16];
    char idxbuf[128];

    for (int i = 0; i < n; i++) {
        if (scanf("%15s", cmdbuf) != 1) break;
        if (strcmp(cmdbuf, "insert") == 0) {
            scanf("%127s", idxbuf);
            scanf("%lld", &valLL);
            Key k;
            makeKey(k, idxbuf, (int32_t)valLL);
            insertKey(k);
        } else if (strcmp(cmdbuf, "delete") == 0) {
            scanf("%127s", idxbuf);
            scanf("%lld", &valLL);
            Key k;
            makeKey(k, idxbuf, (int32_t)valLL);
            deleteKey(k);
        } else if (strcmp(cmdbuf, "find") == 0) {
            scanf("%127s", idxbuf);
            std::string line;
            findIdx(std::string(idxbuf), line);
            outBuf += line;
            outBuf += '\n';
            if (outBuf.size() > (1 << 15)) {
                fwrite(outBuf.data(), 1, outBuf.size(), stdout);
                outBuf.clear();
            }
        }
    }
    if (!outBuf.empty()) fwrite(outBuf.data(), 1, outBuf.size(), stdout);

    closeStorage();
    return 0;
}
