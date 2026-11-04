/**
 * @struct MpiType
 *
 * @brief A mapping from std C++ to MPI types.
 */

#ifndef MpiTypes_H
#define MpiTypes_H

#include <mpi.h>
#include <type_traits>

template<typename T>
struct MpiType;

template<>
struct MpiType<int>
{
    inline static const MPI_Datatype value = MPI_INT;
};

template<>
struct MpiType<float>
{
    inline static const MPI_Datatype value = MPI_FLOAT;
};

template<>
struct MpiType<double>
{
    inline static const MPI_Datatype value = MPI_DOUBLE;
};

template<>
struct MpiType<long>
{
    inline static const MPI_Datatype value = MPI_LONG;
};

#endif  // MpiTypes_H
