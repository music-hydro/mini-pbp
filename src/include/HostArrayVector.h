/**
 * @class HostArrayVector
 *
 * @brief Wraps a contiguous host array as a Tpetra::Vector, taken from MUSIC `src/trilinos/HostArrayVector.h`.
 *
 * When the memory space of the node is host accessible the vector reads and writes the array in place: nothing is
 * allocated and no element is copied. Otherwise Kokkos mirrors the array, copying it in on construction and out on
 * destruction, as the access mode requires.
 *
 * The array must stay alive, and must not be written through any other path, for the lifetime of the wrapper: the
 * vector views it.
 */

#ifndef HostArrayVector_H
#define HostArrayVector_H

#include <cstddef>
#include <Kokkos_Core.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Access.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

/// How the borrowed array is used while the wrapper is alive
enum class HostArrayAccess
{
    ReadOnly,   // Read, never written
    WriteOnly,  // Overwritten, the initial contents are ignored
    ReadWrite,
};

class HostArrayVector
{
  public:
    using Vector = Tpetra::Vector<double>;
    using Map    = Vector::map_type;

    /**
     * @brief Constructor
     *
     * @param[in] map - The map of the vector; the array has one entry per local element.
     * @param[in] data - The array to borrow.
     * @param[in] access - How the array is used.
     */
    HostArrayVector(const Teuchos::RCP<const Map>& map, double data[], const HostArrayAccess access) :
        m_access(access),
        m_mirror(mirrorOf(data, map->getLocalNumElements())),
        m_dual_view(viewOf(data, map->getLocalNumElements(), m_mirror)),
        m_vector(map, m_dual_view)
    {
        // The array carries the input, so the device side is stale. Tpetra syncs it on the first device access, and
        // shares these flags because copying a DualView shares its modified_flags view.
        if (m_access != HostArrayAccess::WriteOnly) { m_dual_view.modify_host(); }
    }

    /**
     * @brief Constructor for an array that is only read.
     *
     * @details Tpetra has no const scalar vector, so the cast is unavoidable; ReadOnly is the contract that makes it
     * safe, and on a host node the array is shared, not copied.
     */
    HostArrayVector(const Teuchos::RCP<const Map>& map, const double data[]) :
        HostArrayVector(map, const_cast<double*>(data), HostArrayAccess::ReadOnly)
    {
    }

    /**
     * @brief Bring the result back into the array; nothing to do on a host node.
     */
    ~HostArrayVector()
    {
        if (m_access != HostArrayAccess::ReadOnly) { m_vector.getLocalViewHost(Tpetra::Access::ReadOnly); }
    }

    HostArrayVector(const HostArrayVector&)             = delete;
    HostArrayVector& operator =(const HostArrayVector&) = delete;

    /**
     * @brief Getter for the vector.
     */
    Vector& vec()
    { return m_vector; }

  private:
    using DualView     = Vector::dual_view_type;
    using HostView     = DualView::t_host;
    using DeviceView   = DualView::t_dev;
    using DeviceMemory = DeviceView::memory_space;

    /**
     * @brief The device side storage, kept as a member so that it outlives the views of it.
     *
     * create_mirror_view gives back its argument when the device memory space is the one of t_host, and allocates
     * otherwise. Kokkos puts t_host in the device memory space whenever that is host accessible, so a node built over
     * Kokkos::SharedSpace needs no allocation and no copy, provided the data really lives in that space (nothing checks
     * it).
     *
     * Warning, only if Trilinos was configured with HAVE_TPETRA_SHARED_ALLOCS (off by default): the memory space of a
     * GPU node is then Kokkos::SharedSpace, and the borrowed memory must have been allocated in that shared address
     * space. If it was not, the first GPU kernel touching it faults.
     */
    static DeviceView mirrorOf(double data[], const size_t n)
    { return Kokkos::create_mirror_view(DeviceMemory(), HostView(data, n, 1)); }

    /**
     * @brief Create a dual view with both sides unmanaged, so that their Kokkos use counts are equal. Tpetra refuses a
     * host access whenever the device view has the higher count, and an unmanaged view counts zero.
     */
    static DualView viewOf(double data[], const size_t n, const DeviceView& mirror)
    { return DualView(DeviceView(mirror.data(), n, 1), HostView(data, n, 1)); }

    const HostArrayAccess m_access;
    DeviceView m_mirror;
    DualView m_dual_view;
    Vector m_vector;
};

#endif  // HostArrayVector_H
