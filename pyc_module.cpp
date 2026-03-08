#include "pyc_module.h"
#include "data.h"
#include <limits>
#include <stdexcept>

namespace {
struct MagicVersion {
    unsigned int magic;
    int major;
    int minor;
    bool unicode;
};

static const MagicVersion kMagicVersions[] = {
    { MAGIC_1_0, 1, 0, false },
    { MAGIC_1_1, 1, 1, false },
    { MAGIC_1_3, 1, 3, false },
    { MAGIC_1_4, 1, 4, false },
    { MAGIC_1_5, 1, 5, false },
    { MAGIC_1_6, 1, 6, false },
    { MAGIC_1_6+1, 1, 6, true },
    { MAGIC_2_0, 2, 0, false },
    { MAGIC_2_0+1, 2, 0, true },
    { MAGIC_2_1, 2, 1, false },
    { MAGIC_2_1+1, 2, 1, true },
    { MAGIC_2_2, 2, 2, false },
    { MAGIC_2_2+1, 2, 2, true },
    { MAGIC_2_3, 2, 3, false },
    { MAGIC_2_3+1, 2, 3, true },
    { MAGIC_2_4, 2, 4, false },
    { MAGIC_2_4+1, 2, 4, true },
    { MAGIC_2_5, 2, 5, false },
    { MAGIC_2_5+1, 2, 5, true },
    { MAGIC_2_6, 2, 6, false },
    { MAGIC_2_6+1, 2, 6, true },
    { MAGIC_2_7, 2, 7, false },
    { MAGIC_2_7+1, 2, 7, true },
    { MAGIC_3_0+1, 3, 0, true },
    { MAGIC_3_1+1, 3, 1, true },
    { MAGIC_3_2, 3, 2, true },
    { MAGIC_3_3, 3, 3, true },
    { MAGIC_3_4, 3, 4, true },
    { MAGIC_3_5, 3, 5, true },
    { MAGIC_3_5_3, 3, 5, true },
    { MAGIC_3_6, 3, 6, true },
    { MAGIC_3_7, 3, 7, true },
    { MAGIC_3_8, 3, 8, true },
    { MAGIC_3_9, 3, 9, true },
    { MAGIC_3_10, 3, 10, true },
    { MAGIC_3_11, 3, 11, true },
    { MAGIC_3_12, 3, 12, true },
    { MAGIC_3_13, 3, 13, true },
    { MAGIC_3_14A1, 3, 14, true },
    { MAGIC_3_14_B3, 3, 14, true },
    { MAGIC_3_14, 3, 14, true },
};
}

void PycModule::setVersion(unsigned int magic)
{
    m_maj = -1;
    m_min = -1;
    m_unicode = false;
    m_exactMagic = false;

    for (size_t i = 0; i < sizeof(kMagicVersions) / sizeof(kMagicVersions[0]); ++i) {
        if (kMagicVersions[i].magic == magic) {
            m_maj = kMagicVersions[i].major;
            m_min = kMagicVersions[i].minor;
            m_unicode = kMagicVersions[i].unicode;
            m_exactMagic = true;
            return;
        }
    }

    if ((magic & 0xFFFF0000U) != 0x0A0D0000U)
        return;

    const MagicVersion* best = NULL;
    unsigned int bestDelta = std::numeric_limits<unsigned int>::max();
    for (size_t i = 0; i < sizeof(kMagicVersions) / sizeof(kMagicVersions[0]); ++i) {
        if (kMagicVersions[i].major < 3)
            continue;

        unsigned int delta = (magic > kMagicVersions[i].magic)
            ? magic - kMagicVersions[i].magic
            : kMagicVersions[i].magic - magic;
        if ((best == NULL) || (delta < bestDelta)
                || ((delta == bestDelta) && (kMagicVersions[i].minor > best->minor))) {
            best = &kMagicVersions[i];
            bestDelta = delta;
        }
    }
    if (best == NULL)
        return;
    // Guard against obviously corrupted headers that only match by nearest distance.
    if (bestDelta > 0x80U)
        return;

    m_maj = best->major;
    m_min = best->minor;
    m_unicode = best->unicode;
}

bool PycModule::isSupportedVersion(int major, int minor)
{
    switch (major) {
    case 1:
        return (minor >= 0 && minor <= 6);
    case 2:
        return (minor >= 0 && minor <= 7);
    case 3:
        return (minor >= 0 && minor <= 14);
    default:
        return false;
    }
}

void PycModule::loadFromFile(const char* filename)
{
    PycFile in(filename);
    if (!in.isOpen()) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }
    unsigned int magic = in.get32();
    setVersion(magic);
    if (!isValid()) {
        fputs("Bad MAGIC!\n", stderr);
        return;
    }
    if (!hasExactMagicMatch()) {
        fprintf(stderr, "Warning: unknown CPython magic 0x%08X, falling back to Python %d.%d\n",
            magic, m_maj, m_min);
    }

    int flags = 0;
    if (verCompare(3, 7) >= 0)
        flags = in.get32();

    if (flags & 0x1) {
        // Optional checksum added in Python 3.7
        in.get32();
        in.get32();
    } else {
        in.get32(); // Timestamp -- who cares?

        if (verCompare(3, 3) >= 0)
            in.get32(); // Size parameter added in Python 3.3
    }

    m_code = LoadObject(&in, this).cast<PycCode>();
}

void PycModule::loadFromMarshalledFile(const char* filename, int major, int minor)
{
    PycFile in (filename);
    if (!in.isOpen()) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }
    if (!isSupportedVersion(major, minor)) {
        fprintf(stderr, "Unsupported version %d.%d\n", major, minor);
        return;
    }
    m_maj = major;
    m_min = minor;
    m_unicode = (major >= 3);
    m_code = LoadObject(&in, this).cast<PycCode>();
}

PycRef<PycString> PycModule::getIntern(int ref) const
{
    if (ref < 0 || (size_t)ref >= m_interns.size())
        throw std::out_of_range("Intern index out of range");
    return m_interns[(size_t)ref];
}

PycRef<PycObject> PycModule::getRef(int ref) const
{
    if (ref < 0 || (size_t)ref >= m_refs.size())
        throw std::out_of_range("Ref index out of range");
    return m_refs[(size_t)ref];
}
