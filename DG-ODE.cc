//
// Created by Jake J. Harmon (harmon@lanl.gov) on 04/24/23.
//


#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>


#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>



#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/base/table_handler.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/fe_interface_values.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/base/utilities.h>

#include <type_traits>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/meshworker/mesh_loop.h>

#include "Quadrature.h"
#include "RandomVariables.h"
#include "RightHandSide.h"

#include <boost/math/tools/roots.hpp>
#include <boost/numeric/odeint.hpp>
#include <memory>

// For the newton solver
#include <deal.II/differentiation/ad.h>

// For transferring solution from old mesh to new
#include <deal.II/numerics/solution_transfer.h>

// Basic error estimation
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

/**
 * For each stochastic injector, which depend on the stochastic coordinates,
 * we have a particular type for the uncertain parameter, e.g., initial
 * conditions boundary conditions, and special types of uncertainty
 * (inter-temporal, which depend additionally on the current time), and regular
 * (which could be, for instance, an uncertain advection speed).
 */
enum UncertainParameterType
{
  INITIAL_CONDITION,
  BOUNDARY_CONDITION,
  INTERTEMPORAL,
  REGULAR,
  NOTHING
};

// Because of petrov-Galerkin method we are using, we save time by simply taking
// the derivatives of the shape functions, so we need these operators
inline double
operator+(const dealii::Tensor<1, 1> &T, const double &t)
{
  return T[0] + t;
}

inline double
operator-(const dealii::Tensor<1, 1> &T, const double &t)
{
  return T[0] - t;
}

inline double
operator+(const double &t, const dealii::Tensor<1, 1> &T)
{
  return t + T[0];
}

inline double
operator-(const double &t, const dealii::Tensor<1, 1> &T)
{
  return t - T[0];
}

inline double
operator*(const double &t, const dealii::Tensor<1, 1> &T)
{
  return t * T[0];
}

inline double
operator*(const dealii::Tensor<1, 1> &T, const double &t)
{
  return t * T;
}

template <int dim, int stochdim>
inline void
extract_subpoint(const dealii::Point<dim + stochdim> &p,
                 dealii::Point<stochdim>             *target)
{
  for (unsigned int i = dim; i < dim + stochdim; ++i)
    target->operator[](i - dim) = p[i];
}

template <int dim, int stochdim>
inline dealii::Point<dim + stochdim>
merge_points(const dealii::Point<dim> &p0, const dealii::Point<stochdim> &p1)
{
  dealii::Point<dim + stochdim> out;
  for (unsigned int i = 0; i < dim; ++i)
    out[i] = p0[i];
  for (unsigned int i = 0; i < stochdim; ++i)
    out[i + dim] = p1[i];

  return out;
}

template <int dim, class Real>
inline Real
dot(const std::array<Real, dim> &p0, const std::array<Real, dim> &p1)
{
  Real out = 0;
  for (unsigned int i = 0; i < dim; ++i)
    out += p0[i] * p1[i];

  return out;
}

template <class VectorClass>
inline VectorClass
abs(const VectorClass &vec)
{
  VectorClass out = vec;
  for (unsigned int i = 0; i < vec.size(); ++i)
    out[i] = std::abs(out[i]);

  return out;
}

template <int dim, class Real>
inline std::array<Real, dim>
product(const std::array<Real, dim> &p0, const Real &scalar)
{
  auto out = p0;
  for (auto &te : out)
    te *= scalar;

  return out;
}

template <int dim, class Real>
inline Real
dot(const std::array<Real, dim>                &p0,
    const std::array<std::vector<double>, dim> &p1,
    const int                                  &index = 0)
{
  Real out = 0;
  for (unsigned int i = 0; i < dim; ++i)
    out += p0[i] * p1[i][index];

  return out;
}
template <int stochdim>
inline dealii::Point<1 + stochdim>
merge_points(const double &p0, const dealii::Point<stochdim> &p1)
{
  dealii::Point<1 + stochdim> out;
  out[0] = p0;
  for (unsigned int i = 0; i < stochdim; ++i)
    out[i + 1] = p1[i];

  return out;
}

template <int dim, int stochdim>
inline dealii::Point<dim + stochdim>
stoch_point_artificial_augmentation(const dealii::Point<stochdim> &p)
{
  dealii::Point<dim + stochdim> out;
  for (unsigned int i = 0; i < stochdim; ++i)
    out[dim + i] = p[i];

  return out;
}

template <int dim>
inline std::array<double, dim>
to_array(const dealii::Point<dim> &p)
{
  std::array<double, dim> out;
  for (unsigned int i = 0; i < dim; ++i)
    out[i] = p[i];

  return out;
}

template <int n_components, class Real>
inline dealii::Point<n_components>
to_point(const std::array<std::vector<Real>, n_components> &data,
         const unsigned int                                &index = 0)
{
  dealii::Point<n_components> out;
  for (unsigned int i = 0; i < n_components; ++i)
    out[i] = data[i][index];

  return out;
}

template <int n_components, class Real>
inline dealii::Point<n_components>
to_point(const std::array<double, n_components> &data)
{
  dealii::Point<n_components> out;
  for (unsigned int i = 0; i < n_components; ++i)
    out[i] = data[i];

  return out;
}

template <class CellPointer, int dim, int stochdim>
inline bool
contains_stochastic_point(const CellPointer             &cell,
                          const dealii::Point<stochdim> &stoch_q_point)
{
  auto vstart = cell->vertex(0);
  auto vend   = cell->vertex(cell->n_vertices() - 1);

  // Check whether the stochastic point is contained in these nodes
  bool valid_cell = true;
  for (unsigned int i = dim; i < dim + stochdim; ++i)
    {
      if (!(vstart[i] <= stoch_q_point[i - dim] &&
            vend[i] >= stoch_q_point[i - dim]))
        {
          valid_cell = false;
          break;
        }
    }
  return valid_cell;
}

namespace Applications
{
  template <int dim, int stochdim>
  struct root_function_hit_value
  {
    root_function_hit_value(const double &threshold_value)
      : threshold_value(threshold_value)
    {}

    std::vector<double>
    operator()(const std::vector<double> &input)
    {
      std::vector<double> out(input.size());
      for (unsigned int i = 0; i < input.size(); ++i)
        out[i] = input[i] - threshold_value;
      return out;
    }

    double
    operator()(const double &input)
    {
      return input - threshold_value;
    }

  private:
    const double threshold_value;
  };

  template <int dim, int stochdim>
  struct root_function_hit_value_deriv
  {
    root_function_hit_value_deriv() = default;

    std::vector<double>
    operator()(const std::vector<double> &value_input,
               const std::vector<double> &value_input_deriv_in_time)
    {
      std::vector<double> out(value_input.size());
      for (unsigned int i = 0; i < value_input.size(); ++i)
        out[i] = value_input_deriv_in_time[i];
      return out;
    }

    double
    operator()(const double &, const double &value_input_deriv_in_time)
    {
      return value_input_deriv_in_time;
    }
  };
} // namespace Applications


namespace Auxiliary
{
  using namespace dealii;

  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> joint_dof_indices;

    void
    reinit(const unsigned int &n_dofs)
    {
      cell_matrix.reinit(n_dofs, n_dofs);
      cell_rhs.reinit(n_dofs);
    }
  };

  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;

    template <class Iterator>
    void
    reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };

  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim>        &mapping,
                const FiniteElement<dim>  &fe,
                const Quadrature<dim>     &quadrature,
                const Quadrature<dim - 1> &quadrature_face,
                const UpdateFlags          update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags interface_update_flags =
                  update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_normal_vectors)
      : fe_values(mapping, fe, quadrature, update_flags)
      , fe_interface_values(mapping,
                            fe,
                            quadrature_face,
                            interface_update_flags)
    {}


    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                            scratch_data.fe_interface_values.get_fe(),
                            scratch_data.fe_interface_values.get_quadrature(),
                            scratch_data.fe_interface_values.get_update_flags())
    {}

    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };
} // namespace Auxiliary


namespace Evaluation
{
  using namespace dealii;

  // Perhaps: loop through all active cells with the intent to identify
  // \emph{all} instances of root function = 0. Flag all cells that possess a
  // sign change in the root function.
  /// TODO: Make suitable for vector input root functions!
  template <int dim, int stochdim>
  class FlagRootCells
  {
  public:
    FlagRootCells(
      std::function<std::vector<double>(const std::vector<double> &)>
                          root_function,
      const unsigned int &component = 0);
    void
    flag_cells(dealii::DoFHandler<dim + stochdim> &dof_handler,
               const dealii::Vector<double>       &solution);

  private:
    std::function<std::vector<double>(const std::vector<double> &)>
                       root_function;
    const unsigned int component;
  };

  template <int dim, int stochdim>
  FlagRootCells<dim, stochdim>::FlagRootCells(
    std::function<std::vector<double>(const std::vector<double> &)>
                        root_function,
    const unsigned int &component)
    : root_function(std::move(root_function))
    , component(component)
  {}

  template <int dim, int stochdim>
  void
  FlagRootCells<dim, stochdim>::flag_cells(
    DoFHandler<dim + stochdim> &dof_handler,
    const Vector<double>       &solution)
  {
    // Reset flags from a previous test
    // dof_handler.get_triangulation().clear_user_flags();

    // Get quadrature. Used to find if sign changes over cell with
    QGaussLobatto<dim + stochdim> quadrature(dof_handler.get_fe().degree + 5);
    // QGauss<dim + stochdim> quadrature(dof_handler.get_fe().degree + 4);
    FEValues<dim + stochdim> fe_out(dof_handler.get_fe(),
                                    quadrature,
                                    update_values);
    const unsigned int       n_quadrature = quadrature.size();
    std::vector<double>      values(n_quadrature);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        // Update FEValues for this cell, and evaluate solution at q points
        fe_out.reinit(cell);
        /// TODO: Generalize to vector input root function
        fe_out[FEValuesExtractors::Scalar(this->component)].get_function_values(
          solution, values);

        // Evaluate at values the root function
        values = root_function(values);

        // Loop through q points, check if sign changes
        bool found_sign_change = false;
        for (unsigned int q_point = 1; q_point < n_quadrature; ++q_point)
          {
            if (values[0] * values[q_point] < 0)
              {
                found_sign_change = true;
                break;
              }
          }
        if (found_sign_change)
          cell->set_user_flag();
      }
  }

  /**
   * New Postprocessing:
   * Divide the stochastic space by a sufficiently accurate quadrature rule of N
   * points/weights in $\mathbb{R}^q$ For each sample in the stochastic space,
   * identify valid cells, over each generating an interpolant over time
   * @tparam dim
   */
  template <int dim, int stochdim, int n_components>
  class EvaluateTrigger
  {
  public:
    EvaluateTrigger(
      std::function<double(const double &)>                 root_function,
      std::function<double(const double &, const double &)> root_function_deriv,
      const std::vector<dealii::Point<stochdim>> &stochastic_quadrature_points);

    DeclException1(PointNotFound,
                   Point<dim>,
                   << "The evaluation point " << arg1
                   << " was not found among the vertices of the present grid.");

    // Loop through all quadrature points. Find cells that are compatible with
    // those quadrature points that have previous been flagged. When a cell is
    // found, interpolate across the cell for a fixed stochastic point. Find the
    // root.
    // TODO: Expand to multiple components
    template <class CellTrackingPointer>
    std::vector<double>
    find_roots(dealii::DoFHandler<dim + stochdim> &dof_handler,
               const dealii::Vector<double>       &solution,
               std::vector<CellTrackingPointer>   &tracker,
               const unsigned int                 &component = 0);

    /**
     * Alternative, more general implementation for the find_roots functionality
     * In this case, component == -1 implies that we will use all components
     * of the forward solution in calculating the root function. Also stores the
     * value of the forward solution (for all vector components, regardless of
     * the setting of component at the time of the event (aka, when the root
     * first occurs)
     */
    template <class CellTrackingPointer>
    std::vector<double>
    find_roots(dealii::DoFHandler<dim + stochdim> &dof_handler,
               const dealii::Vector<double>       &solution,
               std::vector<CellTrackingPointer>   &tracker,
               std::vector<Point<n_components>>   &forward_sol_at_root,
               const int                          &component = -1);

  private:
    std::function<double(const double &)>                 root_function;
    std::function<double(const double &, const double &)> root_function_deriv;
    const std::vector<dealii::Point<stochdim>> &stochastic_quadrature_points;
  };

  template <int dim, int stochdim, int n_components>
  EvaluateTrigger<dim, stochdim, n_components>::EvaluateTrigger(
    std::function<double(const double &)>                 root_function,
    std::function<double(const double &, const double &)> root_function_deriv,
    const std::vector<dealii::Point<stochdim>> &stochastic_quadrature_points)
    : root_function(std::move(root_function))
    , root_function_deriv(std::move(root_function_deriv))
    , stochastic_quadrature_points(stochastic_quadrature_points)
  {}

  template <int dim, int stochdim, int n_components>
  template <class CellTrackingPointer>
  std::vector<double>
  EvaluateTrigger<dim, stochdim, n_components>::find_roots(
    dealii::DoFHandler<dim + stochdim> &dof_handler,
    const dealii::Vector<double>       &solution,
    std::vector<CellTrackingPointer>   &tracker,
    const unsigned int                 &component)
  {
    double              max_root_error = 0;
    std::vector<double> root_locs_out(stochastic_quadrature_points.size(),
                                      std::numeric_limits<double>::max());
    // Loop through given stochastic quadrature points: Trivial to parallelize:
    // no need to worry about race conditions

    // Note: The parent quadrature rule is ALWAYS 1-D, because
    // the interpolant is with respect to time
    QGaussLobatto<1> parent_quadrature(dof_handler.get_fe().degree + 2);

    // Instantiate default mapping
    MappingQ1<dim + stochdim> mapping;

    for (unsigned int stoch_q_point = 0;
         stoch_q_point < stochastic_quadrature_points.size();
         ++stoch_q_point)
      {
        // Loop through all active cells
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->user_flag_set())
              {
                // We have a cell where the root function changes sign.
                // Now find if this cell contains the desired quadrature point

                if (contains_stochastic_point<decltype(cell), dim, stochdim>(
                      cell, stochastic_quadrature_points[stoch_q_point]))
                  {
                    // We have a valid cell. Need to compute interpolant.
                    // Form the set of quadrature points (for a fixed stochastic
                    // point)

                    // Need to convert the stochastic coordinate to unit space
                    auto cell_center = cell->center();
                    for (unsigned int i = dim; i < dim + stochdim; ++i)
                      cell_center[i] =
                        stochastic_quadrature_points[stoch_q_point][i - dim];

                    auto stochastic_point_unit_space =
                      mapping.transform_real_to_unit_cell(cell, cell_center);

                    auto newton_func = [&](const double &t_unit) {
                      // Modify the unit cell quadrature point
                      auto current_quad_point = stochastic_point_unit_space;
                      current_quad_point[0]   = t_unit;
                      // Make the blank quadrature rule
                      const Quadrature<dim + stochdim> quadrature(
                        current_quad_point);

                      FEValues<dim + stochdim> fe_values(dof_handler.get_fe(),
                                                         quadrature,
                                                         update_values |
                                                           update_gradients);
                      fe_values.reinit(cell);

                      std::vector<double>                            val_vec(1);
                      std::vector<Tensor<1, dim + stochdim, double>> deriv_vec(
                        1);
                      fe_values[FEValuesExtractors::Scalar(component)]
                        .get_function_values(solution, val_vec);
                      fe_values[FEValuesExtractors::Scalar(component)]
                        .get_function_gradients(solution, deriv_vec);

                      // Now evaluate the actual root function and its
                      // derivative
                      // Right now constrained to just the first component...
                      double deriv =
                        this->root_function_deriv(val_vec[0], deriv_vec[0][0]);
                      double val = this->root_function(val_vec[0]);

                      return std::make_pair(val, deriv);
                    };

                    boost::uintmax_t max_it = 1000;
                    auto             root_result =
                      boost::math::tools::newton_raphson_iterate(
                        newton_func,
                        0.0, // Start at earliest time (unit space)
                        0.0,
                        1.0,
                        0.95 * std::numeric_limits<double>::digits,
                        max_it);


                    auto check_root_result = [&](const double &test_point) {
                      // Modify the unit cell quadrature point
                      stochastic_point_unit_space[0] = test_point;
                      // Make the blank quadrature rule
                      const Quadrature<dim + stochdim> quadrature(
                        stochastic_point_unit_space);

                      FEValues<dim + stochdim> fe_values(dof_handler.get_fe(),
                                                         quadrature,
                                                         update_values |
                                                           update_gradients);
                      fe_values.reinit(cell);

                      std::vector<double>                            val_vec(1);
                      std::vector<Tensor<1, dim + stochdim, double>> deriv_vec(
                        1);
                      fe_values[FEValuesExtractors::Scalar(component)]
                        .get_function_values(solution, val_vec);
                      fe_values[FEValuesExtractors::Scalar(component)]
                        .get_function_gradients(solution, deriv_vec);

                      // Now evaluate the actual root function and its
                      // derivative
                      double deriv =
                        this->root_function_deriv(val_vec[0], deriv_vec[0][0]);
                      double val = this->root_function(val_vec[0]);

                      return std::make_pair(val, deriv);
                    };

                    // Test if the root function is actually zero at this point
                    auto check_result = check_root_result(root_result);


                    if (std::abs(check_result.first) > 1e-5)
                      {
                        root_result = std::numeric_limits<double>::max();
                        continue;
                      }
                    else
                      {
                        max_root_error =
                          std::max(max_root_error, check_result.first);
                      }

                    // Need to convert the root result from unit space to real
                    // space
                    stochastic_point_unit_space[0] = root_result;


                    const auto new_val = mapping.transform_unit_to_real_cell(
                      cell, stochastic_point_unit_space)[0];
                    if (root_locs_out[stoch_q_point] > new_val)
                      {
                        // Need to update the vectors
                        root_locs_out[stoch_q_point] = new_val;
                        tracker[stoch_q_point]       = cell;
                      }
                  }
              }
          }
      }
    std::cout << "Max root error: " << max_root_error << std::endl;

    // Check how many have odd values for the max time...
    for (unsigned int stoch_q_point = 0;
         stoch_q_point < stochastic_quadrature_points.size();
         ++stoch_q_point)
      {
        if (root_locs_out[stoch_q_point] > 100)
          {
            for (const auto &cell : dof_handler.active_cell_iterators())
              {
                if (cell->user_flag_set())
                  {
                    // We have a cell where the root function changes sign.
                    // Now find if this cell contains the desired quadrature
                    // point

                    if (contains_stochastic_point<decltype(cell),
                                                  dim,
                                                  stochdim>(
                          cell, stochastic_quadrature_points[stoch_q_point]))
                      {
                        cell->set_refine_flag();
                      }
                  }
              }
          }
      }

    return root_locs_out;
  }

  template <int dim, int stochdim, int n_components>
  template <class CellTrackingPointer>
  std::vector<double>
  EvaluateTrigger<dim, stochdim, n_components>::find_roots(
    dealii::DoFHandler<dim + stochdim> &dof_handler,
    const dealii::Vector<double>       &solution,
    std::vector<CellTrackingPointer>   &tracker,
    std::vector<Point<n_components>>   &forward_sol_at_root,
    const int                          &component)
  {
    double              max_root_error = 0;
    std::vector<double> root_locs_out(stochastic_quadrature_points.size(),
                                      std::numeric_limits<double>::max());
    // Loop through given stochastic quadrature points: Trivial to parallelize:
    // no need to worry about race conditions

    // Note: The parent quadrature rule is ALWAYS 1-D, because
    // the interpolant is with respect to time
    QGaussLobatto<1> parent_quadrature(dof_handler.get_fe().degree + 2);

    // Instantiate default mapping
    MappingQ1<dim + stochdim> mapping;

    for (unsigned int stoch_q_point = 0;
         stoch_q_point < stochastic_quadrature_points.size();
         ++stoch_q_point)
      {
        // Loop through all active cells
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->user_flag_set())
              {
                // We have a cell where the root function changes sign.
                // Now find if this cell contains the desired quadrature point

                if (contains_stochastic_point<decltype(cell), dim, stochdim>(
                      cell, stochastic_quadrature_points[stoch_q_point]))
                  {
                    // We have a valid cell. Need to compute interpolant.
                    // Form the set of quadrature points (for a fixed stochastic
                    // point)

                    // Need to convert the stochastic coordinate to unit space
                    auto cell_center = cell->center();
                    for (unsigned int i = dim; i < dim + stochdim; ++i)
                      cell_center[i] =
                        stochastic_quadrature_points[stoch_q_point][i - dim];

                    auto stochastic_point_unit_space =
                      mapping.transform_real_to_unit_cell(cell, cell_center);

                    auto newton_func = [&](const double &t_unit) {
                      // Modify the unit cell quadrature point
                      auto current_quad_point = stochastic_point_unit_space;
                      current_quad_point[0]   = t_unit;
                      // Make the blank quadrature rule
                      const Quadrature<dim + stochdim> quadrature(
                        current_quad_point);

                      FEValues<dim + stochdim> fe_values(dof_handler.get_fe(),
                                                         quadrature,
                                                         update_values |
                                                           update_gradients);
                      fe_values.reinit(cell);

                      std::array<std::vector<double>, n_components> val_vec;
                      std::array<std::vector<Tensor<1, dim + stochdim, double>>,
                                 n_components>
                        deriv_vec;

                      for (unsigned int sol_size = 0; sol_size < n_components;
                           ++sol_size)
                        {
                          val_vec[sol_size].resize(1);
                          deriv_vec[sol_size].resize(1);
                        }

                      for (unsigned int sol_size = 0; sol_size < n_components;
                           ++sol_size)
                        {
                          fe_values[FEValuesExtractors::Scalar(sol_size)]
                            .get_function_values(solution, val_vec[sol_size]);
                          fe_values[FEValuesExtractors::Scalar(sol_size)]
                            .get_function_gradients(solution,
                                                    deriv_vec[sol_size]);
                        }

                      // Now evaluate the actual root function and its
                      // derivative
                      // Right now constrained to just the first component...
                      double deriv =
                        this->root_function_deriv(val_vec[component][0],
                                                  deriv_vec[component][0][0]);


                      double val = this->root_function(val_vec[0]);

                      return std::make_pair(val, deriv);
                    };


                    boost::uintmax_t max_it = 1000;
                    auto             root_result =
                      boost::math::tools::newton_raphson_iterate(
                        newton_func,
                        0.0, // Start at earliest time (unit space)
                        0.0,
                        1.0,
                        0.95 * std::numeric_limits<double>::digits,
                        max_it);


                    auto check_root_result = [&](const double &test_point) {
                      // Modify the unit cell quadrature point
                      stochastic_point_unit_space[0] = test_point;
                      // Make the blank quadrature rule
                      const Quadrature<dim + stochdim> quadrature(
                        stochastic_point_unit_space);

                      FEValues<dim + stochdim> fe_values(dof_handler.get_fe(),
                                                         quadrature,
                                                         update_values);
                      fe_values.reinit(cell);

                      std::array<std::vector<double>, n_components> val_vec;

                      for (unsigned int sol_size = 0; sol_size < n_components;
                           ++sol_size)
                        val_vec[sol_size].resize(1);

                      // Now evaluate the actual root function and its
                      // derivative
                      for (unsigned int sol_size = 0; sol_size < n_components;
                           ++sol_size)
                        {
                          fe_values[FEValuesExtractors::Scalar(sol_size)]
                            .get_function_values(solution, val_vec[sol_size]);
                        }

                      double val = this->root_function(val_vec[component][0]);
                      return std::make_pair(val, to_point(val_vec));
                    };

                    // Test if the root function is actually zero at this point
                    auto check_result = check_root_result(root_result);


                    if (std::abs(check_result.first) > 1e-5)
                      {
                        root_result = std::numeric_limits<double>::max();
                        continue;
                      }
                    else
                      {
                        max_root_error =
                          std::max(max_root_error, check_result.first);
                      }

                    // Need to convert the root result from unit space to real
                    // space
                    stochastic_point_unit_space[0] = root_result;


                    const auto new_val = mapping.transform_unit_to_real_cell(
                      cell, stochastic_point_unit_space)[0];
                    if (root_locs_out[stoch_q_point] > new_val)
                      {
                        // Need to update the vectors
                        root_locs_out[stoch_q_point] = new_val;
                        tracker[stoch_q_point]       = cell;

                        // Now update the vector of the forward solution at this
                        // point
                        forward_sol_at_root[stoch_q_point] =
                          check_result.second;
                      }
                  }
              }
          }
      }
    std::cout << "Max root error: " << max_root_error << std::endl;

    // Check how many have odd values for the max time...
    for (unsigned int stoch_q_point = 0;
         stoch_q_point < stochastic_quadrature_points.size();
         ++stoch_q_point)
      {
        if (root_locs_out[stoch_q_point] > 100)
          {

            for (const auto &cell : dof_handler.active_cell_iterators())
              {
                if (cell->user_flag_set())
                  {
                    // We have a cell where the root function changes sign.
                    // Now find if this cell contains the desired quadrature
                    // point

                    if (contains_stochastic_point<decltype(cell),
                                                  dim,
                                                  stochdim>(
                          cell, stochastic_quadrature_points[stoch_q_point]))
                      {
                        cell->set_refine_flag();
                      }
                  }
              }
          }
      }

    return root_locs_out;
  }

  /**
   * For evaluating sample based quantities at arbitrary locations in the
   * computational domain. For determinstic simulations, could be only samples
   * in time. For stochastic simulations,
   * @tparam dim
   */
  template <int dim>
  class PointValueEvaluation
  {
  public:
    PointValueEvaluation(const dealii::Point<dim> &evaluation_point);



    double
    get_value(const dealii::DoFHandler<dim> &dof_handler,
              const dealii::Vector<double>  &solution) const;

    DeclException1(PointNotFound,
                   Point<dim>,
                   << "The evaluation point " << arg1
                   << " was not found among the vertices of the present grid.");


  private:
    const dealii::Point<dim> evaluation_point;
  };

  template <int dim>
  PointValueEvaluation<dim>::PointValueEvaluation(
    const dealii::Point<dim> &evaluation_point)
    : evaluation_point(evaluation_point)
  {}

  template <int dim>
  double
  PointValueEvaluation<dim>::get_value(
    const dealii::DoFHandler<dim> &dof_handler,
    const dealii::Vector<double>  &solution) const
  {
    double point_value = 1e20;

    bool evaluation_point_found = false;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (!evaluation_point_found)
        for (const auto vertex : cell->vertex_indices())
          if (cell->vertex(vertex) == evaluation_point)
            {
              auto quadrature_out =
                dealii::QGaussLobatto<dim>(2); // Gives just the endpoints...
              FEValues<dim> fe_out(dof_handler.get_fe(),
                                   quadrature_out,
                                   update_values);
              fe_out.reinit(cell);
              std::vector<double> values(2);
              fe_out.get_function_values(solution, values);

              point_value = values[1];


              evaluation_point_found = true;
              break;
            };

    AssertThrow(evaluation_point_found, PointNotFound(evaluation_point));
    return point_value;
  }
} // namespace Evaluation


namespace PostProcessing
{
  /**
   * Integration + density function
   */
  template <int dim, int stochdim>
  double
  integration_on_boundary(
    Auxiliary::ScratchData<dim + stochdim>   &scratch_data,
    const dealii::DoFHandler<stochdim + dim> &dof_handler,
    const dealii::Vector<double>             &primal_solution,
    std::function<double(const double &, const dealii::Point<dim + stochdim> &)>
                        f,
    const unsigned int &component = 0)
  {
    // Loop through cells on terminal boundary
    double return_value = 0.0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const unsigned int face_no : cell->face_indices())
        {
          auto face = cell->face(face_no);
          if (face->boundary_id() == 20)
            {
              // On terminal boundary, evaluate shape functions
              scratch_data.fe_interface_values.reinit(cell, face_no);
              const auto &fe_face_values =
                scratch_data.fe_interface_values.get_fe_face_values(0);

              // Evaluate the solution at the quadrature points along the face
              std::vector<double> forward_vals(
                fe_face_values.n_quadrature_points);
              fe_face_values[dealii::FEValuesExtractors::Scalar(component)]
                .get_function_values(primal_solution, forward_vals);

              for (unsigned int q_index = 0;
                   q_index < fe_face_values.n_quadrature_points;
                   ++q_index)
                {
                  // Get the quadrature point, which is by default in real
                  // space, not unit space
                  const auto quadrature_point =
                    fe_face_values.quadrature_point(q_index);

                  const auto function_val =
                    f(forward_vals[q_index], quadrature_point);

                  return_value += function_val * fe_face_values.JxW(q_index);
                }
            }
        }

    return return_value;
  }

} // namespace PostProcessing

namespace ErrorEstimation
{
  using namespace dealii;

  /**
   * For grabbing temporal data at a particular stochastic coordinate.
   * Creates a set $M$ of cells up to a time of event $t$ belonging to a
   * particular stochastic coordinate.
   *
   * Assumes that we have access to the cell where the event occurs, which
   * we already compute in estimating the time of event at each quadrature
   * point.
   *
   * From that cell, we check its neighbors or active descendants of neighbors.
   * Need to add function for temporal sort of a vector of cell pointers, in the
   * event that we have to cycle through
   * @tparam dim
   * @tparam stochdim
   * @tparam n_components
   */
  template <int dim, int stochdim, int n_components, class CellPointer>
  class SequentialEvaluator
  {
  public:
    // template <class CellPointer>
    SequentialEvaluator(const Point<stochdim> &stoch_point,
                        CellPointer            cell_of_event)
      : stoch_point(stoch_point)
    {
      // Loop through all flagged cells (cells where the event is triggered)
      // There could be multiple flagged cells that own the given stochastic
      // coordinate

      // We start from the event cell
      this->cells_time_descending.push_back(cell_of_event);
      unsigned int current_loc = 0;

      // From this cell, we try to grab the left temporal neighbor (i.e., the
      // 0th face)
      while (true)
        {
          auto temporal_neighbor_try =
            this->cells_time_descending[current_loc]->neighbor(0);

          // If this neighbor does not exist (i.e., we are at boundary, then we
          // are done.
          if (temporal_neighbor_try->level() != -1)
            {
              /*
               * We have three possibilities:
               * 1. The cell neighbor is active (same level)
               * 2. The neighbor is less refined, also still active
               * 3. The neighbor is not active, in which we case we need to scan
               * the child cells and find the correct one The first two
               * possiblities can be handled with the same code
               */
              if (temporal_neighbor_try->is_active())
                {
                  cells_time_descending.push_back(temporal_neighbor_try);
                  ++current_loc;
                }
              else
                {
                  // Need to find the next cell. In this case,
                  // the neighbor cell is more refined. We need to find the
                  // child cell that shares a subface with the current cell, and
                  // also contains the stochastic point only one cell can do
                  // this, unless we made a big mistake and placed a stochastic
                  // point on the boundary of two cells, in which case this will
                  // just pick the first cell

                  // Loop through subfaces until we find the right one.
                  // The right cell DOES exist, so this is safe
                  unsigned int subface_to_check = 0;
                  while (true)
                    {
                      auto test_cell_on_subface =
                        this->cells_time_descending[current_loc]
                          ->neighbor_child_on_subface(
                            0 /* still grabbing subface */, subface_to_check++);
                      if (contains_stochastic_point<
                            decltype(test_cell_on_subface),
                            dim,
                            stochdim>(test_cell_on_subface, stoch_point))
                        {
                          // Append this cell and break
                          cells_time_descending.push_back(test_cell_on_subface);
                          ++current_loc;
                          break;
                        }
                    }
                }
            }
          else
            {
              break;
            }
        }
    }

    std::vector<CellPointer> &
    get_cells()
    {
      return cells_time_descending;
    }

    const std::vector<CellPointer> &
    get_cells() const
    {
      return cells_time_descending;
    }

  private:
    // reference to the stoch point for this seqential evaluator
    const Point<stochdim> &stoch_point;

    std::vector<CellPointer> cells_time_descending;
  };

  template <int dim,
            int stochdim,
            int n_components,
            class StochasticQuadratureRule,
            class CellPointer,
            class RHS,
            class INICON>
  class TimeValueErrorEstimator
  {
  public:
    /**
     * forward_solution = forward solution evaluated at stochcoord
     * when the event occurs
     */
    using AdjointInitializerProblem1 =
      std::function<std::array<double, n_components>()>;
    using AdjointInitializerProblem2 =
      std::function<std::array<double, n_components>(
        const Point<dim + stochdim> & /* stoch coord */,
        const std::array<double, n_components> & /* adjoint_ini_con_1 */,
        const Point<n_components> & /* forward solution */)>;

    using AdjointInitializer =
      std::pair<AdjointInitializerProblem1, AdjointInitializerProblem2>;

    TimeValueErrorEstimator() = default;

    TimeValueErrorEstimator(
      DoFHandler<dim + stochdim>     *dof_handler_ptr,
      const StochasticQuadratureRule &stochastic_quadrature_rule,
      const std::vector<CellPointer> &event_cells,
      const std::vector<double>      &event_times,
      // const std::vector<Point<n_components>> &event_forward_solution_values,
      RHS                  *rhs,
      INICON* inicon,
      const Vector<double> &solution)
      : event_cells(event_cells)
      , event_times(event_times)
      , solution(solution)
    // , event_forward_solution_values(event_forward_solution_values)
    {
      this->stochastic_quadrature_rule = &stochastic_quadrature_rule;
      this->rhs                        = rhs;
      this->inicon = inicon;
      this->dof_handler_ptr            = dof_handler_ptr;
    }
    // Need to have a stochastic quadrature rule. For each stochastic
    // coordinate, we assume there exists some t < T such that time condition is
    // satisfied. Loop through the stochastic quadrature points. For each point
    // in the stochastic space, there exists a set of cells $M$ such that the
    // left temporal boundary is <= t, the time of event. For cell in $M$,
    // integrate e1, e2 from left boundary to {right boundary, or t}. Note that
    // the stochastic quadrature must be fine enough so that a complete rule
    // exists for each cell. Thus, at each stochastic quadrature point, we have
    // a temporal error for all the cells in $M$. Accumulate via numerical
    // integration (with weights associated with each stochastic quadrature
    // point), the contribution over time and stoch space on each cell.

    // For each stochastic point, identify the time to failure.
    // For each stochastic point, with its time to failure, solve a
    // deterministic 1-D adjoint problem. Note that this is for the case where
    // we are interested in expected value of time to failure. If instead we are
    // interested in time to failure of expected value, we can get away with
    // solving one adjoint problem (that is (1 + stochdim)-D).
    Vector<double>
    estimate_error();

    // Need an interpolator struct for each stochastic point that houses
    // a linked list of cells. An ODE solver. Use boost integrate_times(), with
    // times picked so that we perfectly traverse the evaluation at suitable
    // quadrature points. I.e., we start from t and integrate to the cell
    // boundary. The interpolator is then moved to the next cell in the
    // sequence.


    void
    assign_adjoint_initializer(AdjointInitializer adjoint_ode_initializer)
    {
      adjoint_ode_initializer = std::move(adjoint_ode_initializer);
    }

    void
    assign_adjoint_initializer_problem_1(
      AdjointInitializerProblem1 adjoint_ode_initializer_prob_1)
    {
      this->adjoint_ode_initializer.first =
        std::move(adjoint_ode_initializer_prob_1);
    }

    void
    assign_adjoint_initializer_problem_2(
      AdjointInitializerProblem2 adjoint_ode_initializer_prob_2)
    {
      this->adjoint_ode_initializer.second =
        std::move(adjoint_ode_initializer_prob_2);
    }

  private:
    DoFHandler<dim + stochdim>     *dof_handler_ptr;
    const StochasticQuadratureRule *stochastic_quadrature_rule;
    const std::vector<CellPointer> &event_cells;
    const std::vector<double>       event_times;
    const Vector<double>           &solution;
    // const std::vector<Point<n_components>> &event_forward_solution_values;
    RHS *rhs;
    INICON *inicon;


    // For the two adjoint problems (which depend on what QoI) we need to solve
    // for

    AdjointInitializer adjoint_ode_initializer;

    // typedef boost::numeric::odeint::runge_kutta_dopri5<Point<n_components>>
    // error_stepper_type; typedef
    // boost::numeric::odeint::controlled_runge_kutta<error_stepper_type>
    // controlled_stepper;

    typedef std::array<double, n_components> ODESolutionType;
    typedef boost::numeric::odeint::runge_kutta_dopri5<ODESolutionType>
      StateDopri5;

    const double reltol = 1e-10;
    const double abstol = 1e-10;

    struct AdjointODEObserver
    {
      AdjointODEObserver(std::vector<ODESolutionType>* xs, std::vector<double>* times)
      {
        this->xs = xs;
        this->times = times;
      }
      void
      operator()(const ODESolutionType &x, const double &t)
      {
        xs->push_back(x);
        times->push_back(t);
      }
      std::vector<ODESolutionType>* xs;
      std::vector<double>*          times;
    };

    struct AdjointODE
    {
      /**
       *
       * @param parent
       * @param ref_space_q_points : The points in reference space where we need the adjoint solution
       */
      AdjointODE(
        TimeValueErrorEstimator   *parent,
        CellPointer                cell,
        const std::vector<double> &local_solution_coefficients,
        //  const std::vector<Point<dim + stochdim>> &ref_space_q_points,
        const Point<dim + stochdim>
          &real_space_terminal_point, // Note: this is terminal point (in time)
                                      // in REAL TIME. Not ADJOINT TIME.
        const Point<dim + stochdim>
          &real_space_starting_point) // Note: this is the starting point (in
                                      // time) in REAL TIME. Not ADJOINT TIME.
        :                             // ref_space_q_points(ref_space_q_points),
        local_solution_coefficients(local_solution_coefficients), real_space_terminal_point(real_space_terminal_point)
        , real_space_starting_point(real_space_starting_point)
      {
        this->parent = parent;
        this->cell   = cell;
      }
      /**
       * Note that the $t$ specified here is NOT real time. It is reversed time.
       * @param x initial condition
       * @param dxdt x'
       * @param t Reversed time
       */
      void
      operator()(const ODESolutionType &x, ODESolutionType &dxdt, double t)
      {
        // Convert current $t$, (which is in reversed real time) to reference
        // time
        auto donor_real_space_point = real_space_starting_point;
        donor_real_space_point[0] =
          real_space_terminal_point[0] -
          t; // $terminal_time - t$, b/c adjoint is time reversed
        auto donor_ref_space_point =
          mapping.transform_real_to_unit_cell(cell, donor_real_space_point);
        //  const auto t_ref = donor_ref_space_point[0]; //

        Quadrature<dim + stochdim> quadrature(donor_ref_space_point);
        // Now that we have this point, need to evaluate the function at this
        // point
        FEValues<dim + stochdim> fe_point_value(cell->get_fe(),
                                                quadrature,
                                                update_values);
        fe_point_value.reinit(cell);

        // Now we are ready to evaluate the forward solution at the current
        // time.
        std::array<std::vector<double>, n_components> u_value;

        for (unsigned int component = 0; component < n_components; ++component)
          u_value[component].resize(1);


        // Technically, we "should" be using the average of the approximate
        // and exact forward solutions
        for (unsigned int i = 0; i < n_components; ++i)
          fe_point_value[FEValuesExtractors::Scalar(i)]
            .get_function_values_from_local_dof_values(
              local_solution_coefficients, u_value[i]);

        // Now have the solution evaluated at this time
        // Use the RHS function in the parent point to evaluate dxdt using this
        // data
        // TODO: Clean up inefficienies re: to_array and to_point everywhere
        dxdt = to_array(
          parent->rhs->adjoint_value(donor_ref_space_point,
                                     to_point<n_components, double>(x),
                                     to_point<n_components, double>(u_value)));
      }

    private:
      TimeValueErrorEstimator *parent;
      // const std::vector<Point<dim + stochdim>> &ref_space_q_points;
      CellPointer                  cell;
      const std::vector<double>   &local_solution_coefficients;
      const Point<dim + stochdim> &real_space_terminal_point;
      const Point<dim + stochdim> &real_space_starting_point;

      // TODO: Generalize for arbitrary mappings eventually
      MappingQ1<dim + stochdim> mapping;



      // boost::numeric::odeint::controll
    };
  };
  template <int dim,
            int stochdim,
            int n_components,
            class StochasticQuadratureRule,
            class CellPointer,
            class RHS,
            class INICON>
  Vector<double>
  TimeValueErrorEstimator<dim,
                          stochdim,
                          n_components,
                          StochasticQuadratureRule,
                          CellPointer,
                          RHS, INICON>::estimate_error()
  {
    std::cout << "Actually estimating error..." << std::endl;
    Vector<double> error_vec(
      this->dof_handler_ptr->get_triangulation().n_active_cells());
    error_vec = 0;
    Vector<int> n_times_reached(this->dof_handler_ptr->get_triangulation().n_active_cells());
    n_times_reached = 0;

    // For now, just assign the generic mapping
    MappingQ1<dim + stochdim> mapping;
    /* auto                      ode_stepper =
       boost::numeric::odeint::make_controlled<dopri5>(this->abstol,
                                                       this->reltol);*/


    // Loop through each stochastic quadrature point
    // Loop through each of the event cells
    // Loop through each time of event
    for (unsigned int current_stoch_point_index = 0;
         current_stoch_point_index < this->stochastic_quadrature_rule->size();
         ++current_stoch_point_index)
      {
        const double &stoch_weight =
          this->stochastic_quadrature_rule->get_weight(
            current_stoch_point_index);

        // For each stochastic point, we have a vector over partial integrals
        // The length corresponds to the number of cells relevant to this
        // stochastic coordinate, and cannot be known a priori
        std::vector<double> partial_integrals_0, partial_integrals_1;

        // The denominator weighting factor
        // This is simply the QoI evaluated at f(Y(t_c), t_c), i.e.,
        // QoI[f(Y(t_c), t_c)] We evaluate this when on the first cell
        double denominator_weighting = 0;

        // Note that the full error estimate is:
        // $ \sum { partial_integral_0 } / (denominator_weighting + \sum {
        // partial_integral_1 } )


        const Point<stochdim> &current_stoch_point =
          this->stochastic_quadrature_rule->get_abscissa(
            current_stoch_point_index);
        auto current_cell = this->event_cells[current_stoch_point_index];
        const double &current_event_time =
          this->event_times[current_stoch_point_index];

        // Get the vector of cells for this stochastic point
        // std::cout << "Getting SequentialEvaluator..." << std::endl;
        SequentialEvaluator<dim, stochdim, n_components, CellPointer>
                    event_sequence(current_stoch_point, current_cell);
        // std::cout << "Done getting SequentialEvaluator" << std::endl;
        const auto &cell_sequence = event_sequence.get_cells();

        // Handle the integration for the event cell, because we must convert
        // current_event_time to reference temporal space
        // Otherwise, the endpoint in reference temporal space is just 1.0

        /* Point<n_components> adjoint_ini_con = */
        /* initial conditon function */ /* ;
auto                adjoint_ini_con_op = to_array(adjoint_ini_con);*/

        // Need to initialize in the integration loop, depending on where we are
        // at in the simulation to avoid recomputing the forward solution
        // unnecessarily...
        std::array<double, n_components>
          adjoint_ini_con_1, // = this->adjoint_ode_initializer.first(),
          adjoint_ini_con_2;
        /*= this->adjoint_ode_initializer.second(
           stoch_point_artificial_augmentation<dim, stochdim>(
             current_stoch_point),
           adjoint_ini_con_1,
           event_forward_solution_values[current_stoch_point_index]);
*/

        // Convert the end time to reference coorindates
        // Need to convert the stochastic coordinate to unit space
        //        auto cell_center = cell->center();
        //        for (unsigned int i = dim; i < dim + stochdim; ++i)
        //          cell_center[i] =
        //            stochastic_quadrature_points[stoch_q_point][i - dim];
        Point<dim + stochdim> real_space_spawning_point;
        real_space_spawning_point[0] = current_event_time;
        for (unsigned int stoch_index = 0; stoch_index < stochdim;
             ++stoch_index)
          real_space_spawning_point[stoch_index +
                                    1 /* b/c we have 1 time dim*/] =
            current_stoch_point[stoch_index];


        double t_end_real = current_event_time;
        // for each cell: note that the vector of cells in cell_sequence is time
        // reversed

        for (unsigned int time_cell_index = 0;
             time_cell_index < cell_sequence.size();
             ++time_cell_index)
          {
            auto dopri5_0 =
              boost::numeric::odeint::make_controlled<StateDopri5>(this->abstol,
                                                                   this->reltol);
            auto dopri5_1 =
              boost::numeric::odeint::make_controlled<StateDopri5>(this->abstol,
                                                                   this->reltol);
            auto &cell_in_sequence = cell_sequence[time_cell_index];

            const auto dofs_per_cell =
              cell_in_sequence->get_fe().n_dofs_per_cell();
            // Get DoF info
            std::vector<types::global_dof_index> local_dof_indices(
              dofs_per_cell);
            cell_in_sequence->get_dof_indices(local_dof_indices);

            std::vector<double> local_dof_values(dofs_per_cell);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              local_dof_values[i] = this->solution(local_dof_indices[i]);

            // Generate the base quadrature rule
            QGaussLobatto<dim> quadrature(cell_in_sequence->get_fe().degree +
                                          2);


            auto real_space_end_point =
              merge_points(t_end_real, current_stoch_point);
            auto ref_space_end_point =
              mapping.transform_real_to_unit_cell(cell_in_sequence,
                                                  real_space_end_point);
            auto ref_space_start_point = ref_space_end_point;
            ref_space_start_point[0]   = 0.0;
            auto real_space_start_point =
              mapping.transform_unit_to_real_cell(cell_in_sequence,
                                                  ref_space_start_point);

            const double &t_start_real = real_space_start_point[0];

            // Get jacobian for this line integral: in this case, regardless of
            // the curvature of the real cell, it is just a straight line in
            // time: so take (t_end_real - t_start_real)
            double jacobian_real = t_end_real - t_start_real;

            double jacobian_ref =
              ref_space_end_point[0] - ref_space_start_point[0];

            // Now, modify the quadrature rule
            auto donor_points  = quadrature.get_points();
            auto donor_weights = quadrature.get_weights();
            auto donor_weights_face = quadrature.get_weights();

            std::vector<Point<dim + stochdim>> actual_points(
              donor_points.size());
            std::vector<Point<dim + stochdim - 1>> actual_points_face(donor_points.size());

            for (unsigned int i = 0; i < donor_points.size(); ++i)
              {
                donor_weights[i] *= jacobian_real;
                actual_points[i][0] =
                  jacobian_ref * donor_points[i][0] + ref_space_start_point[0];
                for (unsigned int j = 1; j < dim + stochdim; ++j)
                  actual_points[i][j] = ref_space_end_point[j];

                actual_points_face[i][0] = jacobian_ref * donor_points[i][0] + ref_space_start_point[0];
                donor_weights_face[i] *= jacobian_ref;

              }


            Quadrature<dim + stochdim> fe_quadrature(actual_points,
                                                     donor_weights);

            Quadrature<dim + stochdim - 1> fe_face_quadrature(actual_points_face, donor_weights_face);
            // Now initialize the FEValues...
            FEValues<dim + stochdim> fe_values(cell_in_sequence->get_fe(),
                                               fe_quadrature,
                                               update_values |
                                                 update_gradients |
                                                 update_quadrature_points | update_JxW_values);

            FEFaceValues<dim + stochdim> fe_face_values(cell_in_sequence->get_fe(), fe_face_quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);

            fe_values.reinit(cell_in_sequence);
            fe_face_values.reinit(cell_in_sequence, 2);

            auto jxw_vec = fe_values.get_JxW_values();

            // Now need to update the values that we have for the forward
            // solution and the partial derivative in time
            std::array<std::vector<double>, n_components> function_values,
              function_partial_t, function_rhs_values;

            for (unsigned int component = 0; component < n_components;
                 ++component)
              {
                function_values[component].resize(
                  fe_values.n_quadrature_points);
                function_partial_t[component].resize(
                  fe_values.n_quadrature_points);
              }
            for (unsigned int component = 0; component < n_components;
                 ++component)
              {
                auto &fe_view =
                  fe_values[FEValuesExtractors::Scalar(component)];

                fe_view.get_function_values_from_local_dof_values(
                  local_dof_values, function_values[component]);

                // decltype(fe_view)::solution_gradient_type<ADType> test;

                std::vector<typename std::remove_reference<decltype(fe_view)>::
                              type::template solution_gradient_type<double>>
                  temp(fe_values.n_quadrature_points);

                fe_view.get_function_gradients_from_local_dof_values(
                  local_dof_values, temp);
                // Now extract just the partial in time. Could add specific
                // function to deal.ii to make this part easier
                // (get_function_partial...)
                for (unsigned int i = 0; i < temp.size(); ++i)
                  function_partial_t[component][i] = temp[i][0];
              }

            for (unsigned int component = 0; component < n_components;
                 ++component)
              {
                // Now get the values of f(u)
                function_rhs_values[component] =
                  this->rhs->value(fe_values.get_quadrature_points(),
                                   function_values,
                                   component);
              }

            // If we are at the first cell (aka the terminal time), need to
            // genereate the initial conditions for the adjoint problems
            if (time_cell_index == 0)
              {
                adjoint_ini_con_1 = this->adjoint_ode_initializer.first();
                adjoint_ini_con_2 = this->adjoint_ode_initializer.second(
                  stoch_point_artificial_augmentation<dim, stochdim>(
                    current_stoch_point),
                  adjoint_ini_con_1,
                  to_point<n_components, double>(function_values,
                                                 fe_quadrature.size()));

                // Also compute the denominator_weighting
                // Assumes QoI is QoI[u] = <adjoint_ini_con_1, u>
                denominator_weighting =
                  dot<n_components, double>(adjoint_ini_con_1,
                                            function_rhs_values,
                                            int(fe_values.n_quadrature_points -
                                                1));
                // Need to flip sign on adjiont ini con after computing
                // everything else
                adjoint_ini_con_1 =
                  product<n_components, double>(adjoint_ini_con_1, -1.0);
              }

            // At these quadrature points need
            // to evaluate the adjoint problem These are returned in the correct
            // temporal order (i.e., past first, future last) auto
            // std::vector<Point<n_components>> adjoint_states =
            // ...(adjoint_ini_con)
            // The vector needs to be in reversed time, relative to the terminal
            // condition

            // These actual need to be in the correct order, as well... (future
            // first, pre-transformation)
            std::vector<double> adjoint_time_measurement_points(
              quadrature.size());
            for (unsigned int i = 0; i < quadrature.size(); ++i)
              {
                // Need to reverse time: set it to $t - terminal_time$. The
                // initial condition is at terminal time (possibly from the
                // previous (technically future) cell)
                adjoint_time_measurement_points[quadrature.size() - i - 1] =
                  real_space_end_point[0] -
                  mapping.transform_unit_to_real_cell(
                    cell_in_sequence, fe_quadrature.point(i))[0];
              }

            std::vector<double> times_1, times_2;
            std::vector<ODESolutionType> observations_1, observations_2;
            AdjointODEObserver observer_1(&observations_1, &times_1);
            AdjointODEObserver observer_2(&observations_2, &times_2);

            AdjointODE ode_1(this,
                           cell_in_sequence,
                           local_dof_values,
                           real_space_end_point,
                           real_space_start_point);
            AdjointODE ode_2(this,
                              cell_in_sequence,
                              local_dof_values,
                              real_space_end_point,
                              real_space_start_point);

            // Need to solve 2 adjoint problems in order to be able to estimate
            // the error

            // Solve the first adjoint problem
            boost::numeric::odeint::integrate_times(
              dopri5_0,
              ode_1,
              adjoint_ini_con_1,
              adjoint_time_measurement_points.begin(),
              adjoint_time_measurement_points.end(),
              jacobian_real / (double(adjoint_time_measurement_points.size())),
              observer_1);

            // Solve the second adjoint problem
            // Do integral
            boost::numeric::odeint::integrate_times(
              dopri5_1,
              ode_2,
              adjoint_ini_con_2,
              adjoint_time_measurement_points.begin(),
              adjoint_time_measurement_points.end(),
              jacobian_real / (double(adjoint_time_measurement_points.size())),
              observer_2);

          //  const auto &observations_1 = observer_1.xs;
          //  const auto &observations_2 = observer_2.xs;

            if (observations_1.size() == 0 || observations_2.size() == 0)
              {
                std::cout << "Error with ODE solver! No observerations..." << std::endl;
                std::abort();
              }
            if (isnan(observations_1[0][0]) || isnan(observations_2[0][0]))
              {
                std::cout << "nan values for ode solve!" << std::endl;
                std::abort();
              }

            // Now need to get vector of values for the forward solution
            // Also need gradients for the computation


            auto fe_face_values_weights = fe_face_values.get_JxW_values();
            auto fe_face_values_quadrature_points = fe_face_values.get_quadrature_points();

            double partial_int_0 = 0.0;
            double partial_int_1 = 0.0;

            for (unsigned int component = 0; component < n_components;
                 ++component)
              {
            for (unsigned int qpoint = 0; qpoint < observations_1.size();
                 ++qpoint)
              {

                    partial_int_0 +=
                      donor_weights[qpoint] *
                      (observations_1[adjoint_time_measurement_points.size() -
                                      qpoint - 1][component] *
                       (function_rhs_values[component][qpoint] -
                        function_partial_t[component][qpoint]));

                    partial_int_1 +=
                      donor_weights[qpoint] *
                      (observations_2[adjoint_time_measurement_points.size() -
                                      qpoint - 1][component] *
                       (function_rhs_values[component][qpoint] -
                        function_partial_t[component][qpoint]));
                  }
            // Now add the jump term. Depending on the cell, we assign a future term
            // to the neighbor. Otherwise, it goes to this cell.
           // std::cout << "Partial int 0: " << partial_int_0 << ", Partial int 1: " << partial_int_1 << std::endl;
            if (time_cell_index > 0)
              {
                partial_integrals_0[partial_integrals_0.size() - 1] +=
                  observations_1[0][component] *
                  function_values[component][observations_1.size() -
                                             1]; // Add u^- to the future cell
                partial_integrals_1[partial_integrals_1.size() - 1] +=
                  observations_2[0][component] *
                  function_values[component][observations_2.size() -
                                             1]; // Add u^- to the future cell
              }
            if (time_cell_index < cell_sequence.size() - 1)
              {
                partial_int_0 -=
                  observations_1[observations_1.size() - 1][component] *
                  function_values[component][0]; // Add u^+ to this cell
                partial_int_1 -=
                  observations_2[observations_2.size() - 1][component] *
                  function_values[component][0]; // Add u^+ to this cell
              }
            else{
                // Need to get the jump term from the initial condition, because
                // we are now at the temporal starting point
                const auto ini_con_value = this->inicon->value(fe_values.quadrature_point(0), component);
                partial_int_0 +=  observations_1[observations_1.size() - 1][component]*(ini_con_value - function_values[component][0]);
                partial_int_1 +=  observations_2[observations_2.size() - 1][component]*(ini_con_value - function_values[component][0]);
              }
              // std::cout << "Partial int 0: " << partial_int_0 << ", Partial int 1: " << partial_int_1 << std::endl;
              }



            partial_integrals_0.push_back(partial_int_0);
            partial_integrals_1.push_back(partial_int_1);


            // At the end, update adjoint_ini_con, so that we evolve the adjoint
            // solution for the next cell adjoint_ini_con = adjoint_states[0];
            adjoint_ini_con_1 = *(--observations_1.end());
            adjoint_ini_con_2 = *(--observations_2.end());

            // Lastly, if we were on time_cell_index == 0, need to update the
            // end time
            t_end_real = t_start_real;
          }
        // Now compute the refinement indicators from the partial_integrals_0,
        // partial_integrals_1, and denominator_weighting
        {
          std::vector<double> recursive_error_contrib(cell_sequence.size());
          double total_error_num = 0, total_error_den = 0;
          for (unsigned int i = 0; i < cell_sequence.size(); ++i)
            {
              double num_sum = 0, den_sum = 0;
              for (int j = i; j >= 0; --j)
                {
                  num_sum += partial_integrals_0[j];
                  den_sum += partial_integrals_1[j];
                }
              recursive_error_contrib[i] =
                num_sum / (denominator_weighting + den_sum);

              total_error_num += partial_integrals_0[i];
              total_error_den += partial_integrals_1[i];
            }

          // Now compute the telescopic portion, the actual refinement indicator
          ++n_times_reached[cell_sequence[0]->active_cell_index()];
          error_vec(cell_sequence[0]->active_cell_index()) +=
            stoch_weight * recursive_error_contrib[0]; // "First" (i.e., last in
                                                       // time) is already read

          for (unsigned int i = 1; i < cell_sequence.size(); ++i)
            {
              error_vec(cell_sequence[i]->active_cell_index()) +=
                stoch_weight *
                (recursive_error_contrib[i] - recursive_error_contrib[i - 1]);

            }
        }
      }

    return error_vec;
  }

} // namespace ErrorEstimation

namespace cG_and_dG
{
  using namespace dealii;

  template <int dim, int stochdim, int n_components>
  class InitialCondition : public Function<dim + stochdim>
  {
  public:
    // InitialCondition() = default;
    InitialCondition()
      : Function<dim + stochdim>(n_components)
    {}

    /*InitialCondition(std::function<double(const Point<stochdim> &)> f)
    {
      this->stochastic_injector = f;
    }*/

    virtual double
    value(const Point<dim + stochdim> &p,
          const unsigned int           component = 0) const override
    {
      (void)component;
      if (stochdim == 0)
        return 1;
      else
        {
          /*// Extract the portion of
          Point<stochdim> p_stoch;
          extract_subpoint<dim, stochdim>(p, &p_stoch);

          // Now evaluate the stochastic injector at p_stoch
          return this->stochastic_injector(p_stoch);*/

          Point<stochdim> p_stoch;
          extract_subpoint<dim, stochdim>(p, &p_stoch);

          return this->stochastic_injector[component](p_stoch);
        }
    }

    virtual void
    vector_value(const Point<dim + stochdim> &p,
                 Vector<double>              &values) const override
    {
      for (unsigned int comp = 0; comp < n_components; ++comp)
        values(comp) = this->value(p, comp);
    }

    virtual void
    vector_value_list(const std::vector<Point<dim + stochdim>> &ps,
                      std::vector<Vector<double>> &values) const override
    {
      for (unsigned int i = 0; i < ps.size(); ++i)
        for (unsigned int comp = 0; comp < n_components; ++comp)
          values[i](comp) = this->value(ps[i], comp);
    }

    virtual void
    value_list(const std::vector<Point<dim + stochdim>> &ps,
               std::vector<double>                      &values,
               const unsigned int component = 0) const override
    {
      for (unsigned int i = 0; i < ps.size(); ++i)
        values[i] = this->value(ps[i], component);
    }

    void
    set_stochastic_injector(const unsigned int &component,
                            std::function<double(const Point<stochdim> &)> &f)
    {
      this->stochastic_injector[component] = f;
    }

    void
    append_stochastic_injector(
      std::function<double(const Point<stochdim> &)> &f)
    {
      assert(num_assigned_stochastic_injectors < n_components &&
             "Cannot append more injectors than there are components!");

      this->stochastic_injector[num_assigned_stochastic_injectors++] = f;
    }

  private:
    int num_assigned_stochastic_injectors = 0;
    std::function<double(const Point<stochdim> &)>
      stochastic_injector[n_components];
  };

  template <int dim, int stochdim, int n_components, class Real = double>
  class ScalarInitializer : public Function<dim + stochdim>
  {
  public:
    ScalarInitializer() = default;

    virtual double
    value(const Point<dim + stochdim> &p,
          const unsigned int           component = 0) const override
    {
      return std::exp(p[0]);
    }

    virtual void
    value_list(const std::vector<Point<dim + stochdim>> &points,
               std::vector<double>                      &values,
               const unsigned int component = 0) const override
    {
      for (unsigned int i = 0; i < points.size(); ++i)
        values[i] = std::exp(points[i][0]);
    }
  };

  /**
   *
   * @tparam dim The space dimension
   * @tparam stochdim The dimension of the stochastic space
   * @tparam FEType The type of the finite element
   * @tparam n_components The number of components in FEType (i.e., 1 => scalar)
   * @tparam n_uncertain_parameters The number of different parameters controlled by the stochdim stochastic space
   */
  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  class ODESolver
  {
  public:
    typedef Sacado::Fad::DFad<double> ADType;

    // ODESolver();
    ODESolver(
      const unsigned int                         &p            = 1,
      const double                               &max_time     = 5.0,
      RightHandSide<dim, stochdim, n_components> *rhs_function = nullptr);

    void
    run();

    void
    run_newton();

    /**
     * Optional functions for stochdim > 0
     */

    void
    set_stochastic_domain(const Point<stochdim> &a_stoch,
                          const Point<stochdim> &b_stoch)
    {
      this->a_stoch = a_stoch;
      this->b_stoch = b_stoch;

      this->a = merge_points(a_spacetime, a_stoch);
      this->b = merge_points(b_spacetime, b_stoch);
    }

    void
    add_stochastic_injector(const unsigned int &index,
                            std::function<double(const Point<stochdim> &)> f,
                            UncertainParameterType uncertainParameterType =
                              UncertainParameterType::NOTHING)
    {
      this->stochastic_injector[index]       = f;
      this->stochastic_injector_types[index] = uncertainParameterType;
    }

  private:

    unsigned int n_output_files = 0;

    void
    boost_grid(const unsigned int boost_by_level = 1);

    /**
     * FEValuesExtractors. Grabs the components of the solution.
     */
    FEValuesExtractors::Scalar fe_extractors[n_components];


    /**
     * Uncertainty related parameters, etc.
     */

    // The stochastic injector includes ALL possible random perturbations,
    // whether for the initial condition, boundary condition, or something else
    std::function<double(const Point<stochdim> &)>
                           stochastic_injector[n_uncertain_parameters];
    UncertainParameterType stochastic_injector_types[n_uncertain_parameters];

    // Note that, Initial condition uncertainties must be inserted in the order
    // of the components (but need not be inserted first, or consecutively)
    InitialCondition<dim, stochdim, n_components> initial_condition_function;

    void
    distribute_stochastic_injectors();

    /**
     * Regular parameters, etc.
     */
    void
    setup_system();


    // For conventional, linear residual
    void
    assemble_system_galerkin();

    /**
     * For nonlinear problems
     */
    // Main function for nonlinear, deploys the cell/boundary based computations
    void
    assemble_system_galerkin_with_residual_linearization();

    // For the cell volumes
    void
    assemble_cell_terms_for_residual_linearization(
      const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
      Auxiliary::ScratchData<dim + stochdim> &scratch_data,
      Auxiliary::CopyData                    &copy_data);

    // For INTERIOR faces
    void
    assemble_face_terms_for_residual_linearization(
      const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
      const unsigned int &face_index,
      const unsigned int &subface_index,
      const typename DoFHandler<dim + stochdim>::active_cell_iterator
                                             &cell_neighbor,
      const unsigned int                     &neighbor_face_index,
      const unsigned int                     &neighbor_subface_index,
      Auxiliary::ScratchData<dim + stochdim> &scratch_data,
      Auxiliary::CopyData                    &copy_data);

    // For the initial time boundary
    void
    assemble_boundary_terms_for_residual_linearization(
      const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
      const unsigned int                                              &face_no,
      Auxiliary::ScratchData<dim + stochdim> &scratch_data,
      Auxiliary::CopyData                    &copy_data);

    // This function just ensures that the temporal boundary condition is
    // satisfied
    void
    update_initial_guess(Vector<double> &target);

    /**
     *
     */

    void
    solve_system(const double &newton_factor = 0.75);

    /*   void
       solve_galerkin();*/

    void
    postprocess_results(const bool     &for_refinement        = false,
                        Vector<double> *refinement_indicators = nullptr);

    void
    output_results();

    Triangulation<dim + stochdim> triangulation, old_triangulation;
    // FEType                        fe;

    FESystem<dim + stochdim> fe_system;

    DoFHandler<dim + stochdim> dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    // The current solution value
    Vector<double> solution;

    // Vector<double> newton_intermediate;

    // The value of the previous solution (i.e., from the previous mesh, mapped
    // to the current discretization)
    Vector<double> old_solution;

    Vector<double> system_rhs;

    // ExponentialRHS<stochdim> rhs_function;
    RightHandSide<dim, stochdim, n_components> *rhs_function;

    double max_time;

    double time;

    // Extreme points for a hyper-rectangular domain
    Point<dim>            a_spacetime, b_spacetime;
    Point<stochdim>       a_stoch, b_stoch;
    Point<dim + stochdim> a, b;

    // Newton solver parameters
    const double newton_tol = 3e-14;

    // For executing refinement instructions and transferring the old solution
    // to the new mesh
    void
    execute_refinement();

    unsigned int   num_refinements_executed = 0;
    Vector<double> estimated_error_per_cell;
  };

  /*  template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      virtual double
      value(const Point<dim> & */
  /*p*/ /*,
const unsigned int component = 0) const override
{
(void)component;

return 0;
}
};*/

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    ODESolver(const unsigned int                         &p,
              const double                               &max_time,
              RightHandSide<dim, stochdim, n_components> *rhs_function)
    : // fe(p)
    fe_system(FEType(p), n_components)
    , dof_handler(triangulation)
    , max_time(max_time)
  {
    /* For continuous Galerkin, our test functions are actually discontinuous
     * (taking the derivatives of the trial basis functions */
    /*if constexpr (std::is_same<FEType, dealii::FE_Q<dim>>())
      is_petrov = true;
    else
      is_petrov = false;*/

    a_spacetime[0] = 0.0;
    b_spacetime[0] = max_time;

    for (unsigned int i = 0; i < n_components; ++i)
      this->fe_extractors[i] = FEValuesExtractors::Scalar(i);

    this->rhs_function = rhs_function;
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    setup_system()
  {
    this->distribute_stochastic_injectors();
    // GridGenerator::hyper_cube(triangulation, 0, max_time);

    // Generate a hyper rectangular domain
     GridGenerator::hyper_rectangle(triangulation, this->a, this->b);

    // New version with bias mesh
   // std::vector<unsigned int> bias = {1000, 1};
    // GridGenerator::subdivided_hyper_rectangle(triangulation, bias, this->a, this->b);

    triangulation.refine_global(2);

    for (const auto &cell : triangulation.cell_iterators())
      for (const auto &face : cell->face_iterators())
        {
          const auto center = face->center();
          // If located at temporal boundary
          if (std::fabs(center(0)) < 1e-14 && face->at_boundary())
            {
              face->set_boundary_id(10);
            }
          else if (std::fabs(center(0) - max_time) < 1e-14 &&
                   face->at_boundary())
            face->set_boundary_id(20); //
        }

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs(fe_system);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;


    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    /* if (is_petrov)
       DoFTools::make_sparsity_pattern(dof_handler, dsp);
     else*/
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Apply boundary conditions
    constraints.clear();

    constraints.close();
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    update_initial_guess(Vector<double> & /* target */)
  {
    /*  // Just uses interpolate at the boundary with the InitialCondition to
      // constrain the solution
      AffineConstraints<double> constraints_bc;

      // std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler,
                                               10,
                                               Functions::ZeroFunction<dim +
      stochdim>(n_components),
                                             //  initial_condition_function,
                                               constraints_bc);

      // Other test

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler, 10,
      initial_condition_function, boundary_values);

      std::cout << "boundary_values_map (size =  " << boundary_values.size() <<
      "):"; for (auto bv_it = boundary_values.begin(); bv_it !=
      boundary_values.end(); ++bv_it)
        {
          std::cout << " " << bv_it->second;
        }
      std::cout << std::endl;

      constraints_bc.close();

      // Test initial condition function
      Point<dim + stochdim> test(0.0, 0.5);
      std::cout << "Evaluation of initial condition: " <<
      initial_condition_function.value(test, 0) << std::endl;

      // Updating initial guess for boundary condition
      std::cout << "Before apply update: " << target << std::endl;
      constraints_bc.distribute(target);
      std::cout << "********************" << std::endl;
      std::cout << "After apply update: " << target << std::endl;*/
  }

  /**
   * As the continuous Galerkin method results in a non-symmetric system matrix,
   * a typical CG solver is not suitable. Intead, we use a GMRES/direct LU
   * solver...
   * @tparam dim
   * @tparam FEType
   */
  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    solve_system(const double &newton_factor)
  {
    Vector<double> newton_update(this->dof_handler.n_dofs());
    newton_update = 0;

    SparseDirectUMFPACK A_direct;

    A_direct.initialize(system_matrix);
    A_direct.vmult(newton_update, system_rhs);

    // Now have newton update, add to solution...

    this->solution.add(newton_factor, newton_update);
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    output_results()
  {
    DataOut<dim + stochdim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(
      solution,
      "solution",
      DataOut<dim + stochdim>::DataVectorType::type_dof_data);

    Vector<double> user_flag_vec(
      dof_handler.get_triangulation().n_active_cells());
    unsigned int index = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      user_flag_vec[index++] = cell->user_flag_set();

    data_out.add_data_vector(
      user_flag_vec,
      "user_flag",
      DataOut<dim + stochdim>::DataVectorType::type_cell_data);

    data_out.add_data_vector(
      abs(this->estimated_error_per_cell),
      "error",
      DataOut<dim + stochdim>::DataVectorType::type_cell_data);

    Vector<double> cell_indices(
      dof_handler.get_triangulation().n_active_cells()),
      cell_levels(dof_handler.get_triangulation().n_active_cells());
    index = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_indices[index]  = cell->index();
        cell_levels[index++] = cell->level();
      }
    data_out.add_data_vector(
      cell_indices,
      "index",
      DataOut<dim + stochdim>::DataVectorType::type_cell_data);
    data_out.add_data_vector(
      cell_levels,
      "level",
      DataOut<dim + stochdim>::DataVectorType::type_cell_data);

    data_out.build_patches();

    const std::string filename = "plotting/solution_" + std::to_string(this->n_output_files) + ".vtu";
    ++n_output_files;

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::run()
  {
    setup_system();



    /* if (is_petrov)
       this->assemble_system_petrov();
     else*/
    this->assemble_system_galerkin();

    solve_system();
    postprocess_results();
    output_results();
    old_solution = solution;
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    run_newton()
  {
    const bool         boost_initial_iterations = true;
    const unsigned int num_boost_times          = 1;
    const unsigned int num_refinement_times = 5;
    // TODO: Set to only do once, or include adaptivity here...
    setup_system();


    // First time, need to ensure B.C. satisifed for initial guess
    this->update_initial_guess(this->solution);
    this->solution = 0;

    // VectorTools::interpolate(dof_handler, ScalarInitializer<dim, stochdim,
    // n_components>(), this->solution); std::cout << "Solution: " <<
    // this->solution << std::endl;

    this->num_refinements_executed = 0;
    const unsigned int max_num_refinements =
      num_boost_times + num_refinement_times + 1; // + 2 ; // add in the number of extra times to actual
                       // refine as opposed to just boost

    while (/*true*/    /* adaptivity tolerance not satisfied*/
           num_refinements_executed < max_num_refinements) // Iteration loop for
                                                           // newton method
      {

        // this->solution = 0;
        // Need to add initialization of old_solution if we aren't here post
        // adaptivity... If this is the first time calling, then we generate a
        // guess with the "initial condition" and zeros elsewhere Otherwise, we
        // just enforce that the boundary condition is still satisfied...

        unsigned int n_newton_iterations  = 0;
        double       current_newton_error = 0;
        do
          {
            // Reset matrices
            system_matrix = 0;
            system_rhs    = 0;

            // Assemble systems for the AD linearized residual
            this->assemble_system_galerkin_with_residual_linearization();

            if (num_refinements_executed == 0)
              solve_system();
            else
              solve_system(1.0 /* newton factor */);

            // Set the tol to the l2_norm of the residual
            current_newton_error = system_rhs.l2_norm();

            // TODO: Maybe old solution should be converged solution for prev
            // mesh?
            //  old_solution = solution;
            std::cout << "Current newton error: " << current_newton_error
                      << std::endl;
            ++n_newton_iterations;
          }
        while (std::fabs(current_newton_error) >
               this->newton_tol /* newton tolerance not satisifed */
               /*|| n_newton_iterations++ < 5*/);

        std::string msg_line = "******* DONE WITH ROOT SOLVER FOR MESH " +
                               std::to_string(num_refinements_executed) + ": " +
                               std::to_string(n_newton_iterations) +
                               " ITERATIONS REQUIRED *******";
        std::string border_line = std::string(msg_line.size(), '*');
        std::cout << border_line << std::endl
                  << msg_line << std::endl
                  << border_line << std::endl;

        // Now that we have a converged solution, need to estimate error, flag
        // refinements flag refinements here



        if (boost_initial_iterations &&
            num_refinements_executed < num_boost_times &&
            max_num_refinements > 1 && num_refinements_executed < max_num_refinements - 1)
          this->boost_grid(1);
        else
          {
            /////////////////////////////////////////////////
            // Old version based only on KellyErrorEstimator:
            /////////////////////////////////////////////////

            // Vector<float> estimated_error_per_cell(
            //   triangulation.n_active_cells());

            // KellyErrorEstimator<dim + stochdim>::estimate(
            //   dof_handler,
            //   QGauss<dim + stochdim - 1>(fe_system.degree + 1),
            //   {},
            //   solution,
            //   estimated_error_per_cell);

            ////////////////////////////////////////////////////
            // New version based on time valued error estimation
            ////////////////////////////////////////////////////
            std::cout << "==========================================================" << std::endl;
            std::cout << "                STARTING ERROR ESTIMATION                 " << std::endl;

            this->estimated_error_per_cell.reinit(
              triangulation.n_active_cells());

            this->output_results();

            postprocess_results(true /* for refinement */,
                                &estimated_error_per_cell);

            GridRefinement::refine_and_coarsen_fixed_number(
              triangulation, abs(estimated_error_per_cell), 0.5, 0.03);



            // After flagging refinements, execute and transfer solution over
            if (num_refinements_executed < max_num_refinements - 1)
              {
                std::cout << "Refining based on error estimates..." << std::endl;
                this->execute_refinement();
                std::cout << "DONE!" << std::endl;
              }

            std::cout << "                DONE WITH ERROR ESTIMATION                " << std::endl;
            std::cout << "==========================================================" << std::endl;

          }

        // Since for now, testing without adaptivity

        ++num_refinements_executed;
        // break;
      }

    postprocess_results();
    output_results();
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    assemble_system_galerkin()
  {
    // RightHandSide<dim + stochdim> rhs_function;



    // Initializ

    // Create the worker for the cell contributions
    const auto cell_worker =
      [&](const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
          Auxiliary::ScratchData<dim + stochdim> &scratch_data,
          Auxiliary::CopyData                    &copy_data) {
        const unsigned int dofs_per_cell =
          scratch_data.fe_values.get_fe().n_dofs_per_cell();
        copy_data.reinit(cell, dofs_per_cell);
        scratch_data.fe_values.reinit(cell);

        const FEValues<dim + stochdim> &fe_values = scratch_data.fe_values;
        {
          // Now loop through test functions
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe_system.system_to_component_index(i).first;
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const unsigned int component_j =
                    fe_system.system_to_component_index(j).first;
                  for (unsigned int q_index = 0;
                       q_index < fe_values.n_quadrature_points;
                       ++q_index)
                    {
                      // Integrate the cell term, i.e., \int_T_k < U' - f(U),
                      // \psi > dt
                      copy_data.cell_matrix(i, j) +=
                        (fe_values.shape_grad_component(j, q_index, component_j)
                           [0] - // Only want the partial derivative in time. No
                                 // new physics from making stochastic!

                         fe_values.shape_value_component(
                           j, // RHS for exponential ODE
                           q_index,
                           component_j)

                           ) *
                        fe_values.shape_value_component(i,
                                                        q_index,
                                                        component_i) *
                        fe_values.JxW(q_index);
                    }
                }
            }
        }
      };

    // Now we have to deal with the cell(s) on the left boundary
    const auto boundary_worker =
      [&](const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
          const unsigned int                     &face_no,
          Auxiliary::ScratchData<dim + stochdim> &scratch_data,
          Auxiliary::CopyData                    &copy_data) {
        // In this case (recall that we are on the "left"-most boundary), we
        // have a right-hand-side term result from the initial condition a.k.a.
        // $U_0^{-}$ from the limit from the left. We also have the
        // approximation from this cell (limit from the right)
        //
        // Get the face
        const auto &face = cell->face(face_no);
        if (face->boundary_id() == 10)
          {
            /*InitialCondition<dim, stochdim> initial_condition_function(
              this->stochastic_injector);*/
            // Get the necessary face value structures
            scratch_data.fe_interface_values.reinit(cell, face_no);
            const auto &fe_face_values =
              scratch_data.fe_interface_values.get_fe_face_values(0);

            const unsigned int n_facet_dofs =
              fe_face_values.get_fe().n_dofs_per_cell();


            // We only perform these operations on this face
            // First, settle the contribution to the right-hand-side

            // Since we are in 1-D, these aren't actually integrals, but rather
            // just multiplications


            for (unsigned int i = 0; i < n_facet_dofs; ++i)
              {
                const unsigned int component_i =
                  fe_system.system_to_component_index(i).first;
                for (unsigned int q_index = 0;
                     q_index < fe_face_values.n_quadrature_points;
                     ++q_index)
                  {
                    /*
                     * The full inner product is: $ \langle u^+ - u^-,\, v^+
                     * \rangle$, Which results in $\langle u^',\, v^+ \rangle$
                     * being sent to the RHS and $\langle u^+,\, v^+ \rangle$
                     * assigned to the cell matrix...
                     */
                    {
                      copy_data.cell_rhs(i) +=
                        fe_face_values.shape_value(i, q_index) *
                        initial_condition_function.value(
                          fe_face_values.quadrature_point(q_index),
                          component_i) *
                        fe_face_values.JxW(q_index);
                    }

                    for (unsigned int j = 0; j < n_facet_dofs; ++j)
                      copy_data.cell_matrix(i, j) +=
                        fe_face_values.shape_value(i, q_index) *
                        fe_face_values.shape_value(j, q_index) *
                        fe_face_values.JxW(q_index);
                  }
              }
          }
      };

    // This is where the bulk of the work making the dG method work occurs. We
    // have to accumulate the contributions at the interfaces between two cells
    // NOT at the boundaries
    const auto face_worker =
      [&](const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
          const unsigned int &face_index,
          const unsigned int &subface_index,
          const typename DoFHandler<dim + stochdim>::active_cell_iterator
                                                 &cell_neighbor,
          const unsigned int                     &neighbor_face_index,
          const unsigned int                     &neighbor_subface_index,
          Auxiliary::ScratchData<dim + stochdim> &scratch_data,
          Auxiliary::CopyData                    &copy_data) {
        // Evaluate <u_{n-1}^+ - u_{n-1}^-, v^+>
        // Need to grab the values from the two cells at the faces...
        FEInterfaceValues<dim + stochdim> &fe_iv =
          scratch_data.fe_interface_values;
        fe_iv.reinit(cell,
                     face_index,
                     subface_index,
                     cell_neighbor,
                     neighbor_face_index,
                     neighbor_subface_index);

        const auto &q_points = fe_iv.get_quadrature_points();

        copy_data.face_data.emplace_back();
        Auxiliary::CopyDataFace &copy_data_face = copy_data.face_data.back();

        const unsigned int n_dofs        = fe_iv.n_current_interface_dofs();
        copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

        copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);
        const std::vector<Tensor<1, dim + stochdim>> &normals =
          fe_iv.get_normal_vectors();
        // B/c the fe_system is primitive, namely only one non-zero
        // component for each degree of freedom, we can skip checking
        // for each component separately
        for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
          for (unsigned int i = 0; i < n_dofs; ++i)
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                /*
                 * It may appear here that the integrals aren't correctly
                 * accounting for the discontinuity of the shape functions
                 * (i.e., we evaluate the jump just for the shape function j).
                 * However, we simply have that one side of the shape function
                 * will be zero...
                 */
                /*
                 * The ``plus" value is from the left side (Ref Coord 0),
                 * ``minus" from the right side side (Ref Coord 1):
                 * [----Cell X ----(-)][(+)----Cell Y----]
                 */
                /* std::cout << "Jump of shape function: " << j << " with value
                   "
                           << fe_iv.jump_in_shape_values(j */
                /* u^+ - u^- */ /*,
             qpoint)
<< std::endl;*/


                copy_data_face.cell_matrix(i, j) -=
                  fe_iv.shape_value(normals[qpoint][0] <
                                      0 // Just using false doesn't work in the
                                        // case of N-irregular meshes, N > 0
                                    /* v^+        */,
                                    i,
                                    qpoint) *
                  fe_iv.jump_in_shape_values(j
                                             /*u^+ - u^-  */,
                                             qpoint) *
                  fe_iv.JxW(qpoint);
              }
      };
    QGauss<dim + stochdim>     quadrature(fe_system.tensor_degree() + 1);
    QGauss<dim - 1 + stochdim> quadrature_face(fe_system.tensor_degree() + 1);

    FEValues<dim + stochdim> fe_values(fe_system,
                                       quadrature,
                                       update_values | update_gradients |
                                         update_JxW_values |
                                         update_quadrature_points);


    Auxiliary::ScratchData<dim + stochdim> scratch_data(fe_values.get_mapping(),
                                                        this->fe_system,
                                                        quadrature,
                                                        quadrature_face);
    Auxiliary::CopyData                    copy_data;

    const AffineConstraints<double> constraints;

    const auto copier = [&](const Auxiliary::CopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             system_rhs);

      for (auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.joint_dof_indices,
                                                 system_matrix);
        }
    };

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    assemble_system_galerkin_with_residual_linearization()
  {
    QGauss<dim + stochdim>     quadrature(fe_system.tensor_degree() + 1);
    QGauss<dim - 1 + stochdim> quadrature_face(fe_system.tensor_degree() + 1);

    FEValues<dim + stochdim> fe_values(fe_system,
                                       quadrature,
                                       update_values | update_gradients |
                                         update_JxW_values |
                                         update_quadrature_points);


    Auxiliary::ScratchData<dim + stochdim> scratch_data(fe_values.get_mapping(),
                                                        this->fe_system,
                                                        quadrature,
                                                        quadrature_face);
    Auxiliary::CopyData                    copy_data;

    const AffineConstraints<double> constraints;

    const auto copier = [&](const Auxiliary::CopyData &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             system_rhs);

      for (auto &cdf : c.face_data)
        {
          constraints.distribute_local_to_global(cdf.cell_matrix,
                                                 cdf.cell_rhs,
                                                 cdf.joint_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }
    };

    const auto cell_worker =
      [&](const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
          Auxiliary::ScratchData<dim + stochdim> &scratch_data,
          Auxiliary::CopyData                    &copy_data) {
        return this->assemble_cell_terms_for_residual_linearization(
          cell, scratch_data, copy_data);
      };

    const auto boundary_worker =
      [&](const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
          const unsigned int                     &face_no,
          Auxiliary::ScratchData<dim + stochdim> &scratch_data,
          Auxiliary::CopyData                    &copy_data) {
        return this->assemble_boundary_terms_for_residual_linearization(
          cell, face_no, scratch_data, copy_data);
      };

    const auto face_worker =
      [&](const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
          const unsigned int &face_index,
          const unsigned int &subface_index,
          const typename DoFHandler<dim + stochdim>::active_cell_iterator
                                                 &cell_neighbor,
          const unsigned int                     &neighbor_face_index,
          const unsigned int                     &neighbor_subface_index,
          Auxiliary::ScratchData<dim + stochdim> &scratch_data,
          Auxiliary::CopyData                    &copy_data) {
        return this->assemble_face_terms_for_residual_linearization(
          cell,
          face_index,
          subface_index,
          cell_neighbor,
          neighbor_face_index,
          neighbor_subface_index,
          scratch_data,
          copy_data);
      };

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    postprocess_results(const bool     &for_refinement,
                        Vector<double> *refinement_indicators)
  {
    // Clear any flags from a previous postprocessing run
    this->triangulation.clear_user_flags();

    // TODO: Make set by input prm file
    auto rv = Stochastic::UniformRV<double>(0.0, 1.0);


    if (!for_refinement)
      {
        auto expected_value_of =
          [&](const double &u, const Point<dim + stochdim> &real_cell_point) {
            Point<stochdim> stoch_part;
            extract_subpoint<dim, stochdim>(real_cell_point, &stoch_part);
            auto temp = rv.evaluate_pdf(stoch_part[0]);
            return u * temp;
          };

        QGauss<dim + stochdim>     quadrature(fe_system.tensor_degree() + 1);
        QGauss<dim - 1 + stochdim> quadrature_face(fe_system.tensor_degree() +
                                                   1);

        FEValues<dim + stochdim> fe_values(fe_system,
                                           quadrature,
                                           update_values | update_gradients |
                                             update_JxW_values |
                                             update_quadrature_points);


        Auxiliary::ScratchData<dim + stochdim> scratch_data(
          fe_values.get_mapping(),
          this->fe_system,
          quadrature,
          quadrature_face);

        auto expected_value =
          PostProcessing::integration_on_boundary<dim, stochdim>(
            scratch_data, this->dof_handler, solution, expected_value_of);
        std::cout << std::setprecision(20)
                  << "Expected value dG: " << expected_value << std::endl;

        /*InitialCondition<dim, stochdim> initial_condition_function(
          this->stochastic_injector);*/

        auto boost_exact_expectation = [&](const double &stoch_val) {
          return std::exp(this->max_time) *
                 initial_condition_function.value(
                   Point<dim + stochdim>(0, stoch_val)) *
                 rv.evaluate_pdf(stoch_val);
        };
        std::cout << "Expected value exact: "
                  << boost::math::quadrature::gauss<double, 100>::integrate(
                       boost_exact_expectation, rv.a, rv.b)
                  << std::endl;
      }
    // Generate a stochastic quadrature structure
    Point<stochdim, int> n_subdivs(128);
    StochasticIntegration::TensorProductRule<stochdim,
                                             quadrature::QGauss<4, double>>
      tensor_rule(a_stoch, b_stoch, n_subdivs);

    // From the tensor rule, get the stochastic quadrature points...
    auto stoch_q_points = tensor_rule.get_abscissas();
    std::vector<decltype((dof_handler.begin_active()))> cell_pointers(
      tensor_rule.size()); // For each quadrature point, assign the cell where
                           // the event is triggered
    /* std::vector<Point<n_components>> event_forward_solution_values(
       tensor_rule.size()); // For each quadrature point, get the forward
                            // solution value at this point as well*/

    // Actually perform the checks
    Evaluation::FlagRootCells<dim, stochdim> flagger(
      Applications::root_function_hit_value<dim, stochdim>(-10.0));
    flagger.flag_cells(dof_handler, solution);
    // Feed through TriggerEvaluator
    Evaluation::EvaluateTrigger<dim, stochdim, n_components> time_to_threshold(
      Applications::root_function_hit_value<dim, stochdim>(-10.0),
      Applications::root_function_hit_value_deriv<dim, stochdim>(),
      stoch_q_points);

    auto roots_at_stoch_points = time_to_threshold.find_roots(this->dof_handler,
                                                              this->solution,
                                                              cell_pointers);
    if (for_refinement)
      {
        assert(
          refinement_indicators != nullptr &&
          "In order to use for refinement, refinement_indicators must be a valid Vector<double> pointer!");
        ErrorEstimation::TimeValueErrorEstimator<
          dim,
          stochdim,
          n_components,
          decltype(tensor_rule),
          decltype(dof_handler.begin_active()),
          RightHandSide<dim, stochdim, n_components>, InitialCondition<dim, stochdim, n_components>>
          error_estimator(&dof_handler,
                          tensor_rule,
                          cell_pointers,
                          roots_at_stoch_points,
                          this->rhs_function, &(this->initial_condition_function),
                          this->solution);

        // Based on the QoI, need to assign the initial conditions for the
        // adjoint problems
        // TODO: Make set by input file
        error_estimator.assign_adjoint_initializer_problem_1([]() {
          std::array<double, n_components> out;
          for (auto &te : out)
            te = 0;
          out[0] = 1.0; // picks out first component
          return out;
        });

        error_estimator.assign_adjoint_initializer_problem_2(
          [&](const Point<dim + stochdim>            &stochcoord,
              const std::array<double, n_components> &adjoint_ini_con_1,
              const Point<n_components>              &forward_solution) {
            return this->rhs_function->grad_multiplied(stochcoord,
                                                       adjoint_ini_con_1,
                                                       forward_solution);
          });

        *refinement_indicators = error_estimator.estimate_error();
      }

    std::cout << "Stoch point values: " << std::endl;
    for (unsigned int stoch_q_point = 0; stoch_q_point < stoch_q_points.size();
         ++stoch_q_point)
      {
        // At every point in the stochastic space need to multiply by the
        // respective probability density of the uncertainty
        roots_at_stoch_points[stoch_q_point] *=
          rv.evaluate_pdf(stoch_q_points[stoch_q_point][0]);
      }

    // Now have all roots at the stoch quadrature points. Integrate:
    auto expected_time_to_value = tensor_rule.integrate(roots_at_stoch_points);

    std::cout << "******************" << std::endl;
    std::cout << "Expected time to value (FEM): " << expected_time_to_value
              << std::endl;

    // For simple exponetial problem
    // std::cout << "Expected time to value (Exact): " << log(60.0 / 4.0) + 1.0
       //       << std::endl;

    std::cout << "Expected time to value (Reference): " << 1.2063412712107268732 << std::endl;
    std::cout << "Total error: " << 1.2063412712107268732 - expected_time_to_value << std::endl;

    double error_indicator_sum = 0;
    for (const auto& te : this->estimated_error_per_cell)
      error_indicator_sum += te;

    std::cout << "Sum of error indicators: " << error_indicator_sum << std::endl;

  }
  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    distribute_stochastic_injectors()
  {
    // Loop through the stochastic injectors, and assign them where they need to
    // go
    for (unsigned int i = 0; i < n_uncertain_parameters; ++i)
      {
        switch (this->stochastic_injector_types[i])
          {
            case UncertainParameterType::INITIAL_CONDITION:
              {
                // Append the initial condition injector
                this->initial_condition_function.append_stochastic_injector(
                  this->stochastic_injector[i]);
              }
              break;
            default:
              {
                std::cout
                  << "ERROR! Every uncertain parameter must have  a type"
                  << std::endl;
                std::abort();
              }
          }
      }
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    assemble_cell_terms_for_residual_linearization(
      const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
      Auxiliary::ScratchData<dim + stochdim> &scratch_data,
      Auxiliary::CopyData                    &copy_data)
  {
    /*const auto cell_worker =
      [](const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
          Auxiliary::ScratchData<dim + stochdim> &scratch_data,
          Auxiliary::CopyData                    &copy_data) */
    {
      const unsigned int dofs_per_cell =
        scratch_data.fe_values.get_fe().n_dofs_per_cell();
      copy_data.reinit(cell, dofs_per_cell);
      scratch_data.fe_values.reinit(cell);
      FEValues<dim + stochdim> &fe_values = scratch_data.fe_values;


      // Need the AD equivalents for the DoF values at the interface
      std::vector<ADType> dof_values(dofs_per_cell);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // Grab the actual value for this variable
          dof_values[i] = this->solution(copy_data.local_dof_indices[i]);
          // Let the AD system know we need partials wrt this
          dof_values[i].diff(i, dofs_per_cell);
        }

      // Now, we need at each quadrature point the jump in the function values
      // std::vector<ADType> function_values[n_components],
      // function_partial_t[n_components], f_values[n_components];

      std::array<std::vector<ADType>, n_components> function_values,
        function_partial_t, f_values;
      for (unsigned int component = 0; component < n_components; ++component)
        {
          function_values[component].resize(fe_values.n_quadrature_points);
          function_partial_t[component].resize(fe_values.n_quadrature_points);
          f_values[component].resize(fe_values.n_quadrature_points);
        }

      // Need to generate u_t - f(u)
      for (unsigned int component = 0; component < n_components; ++component)
        {
          auto &fe_view = fe_values[this->fe_extractors[component]];

          fe_view.get_function_values_from_local_dof_values(
            dof_values, function_values[component]);

          // decltype(fe_view)::solution_gradient_type<ADType> test;

          std::vector<typename std::remove_reference<
            decltype(fe_view)>::type::template solution_gradient_type<ADType>>
            temp(fe_values.n_quadrature_points);

          fe_view.get_function_gradients_from_local_dof_values(dof_values,
                                                               temp);

          // Now extract just the partial in time. Could add specific function
          // to deal.ii to make this part easier (get_function_partial...)
          for (unsigned int i = 0; i < temp.size(); ++i)
            function_partial_t[component][i] = temp[i][0];
        }

      for (unsigned int component = 0; component < n_components; ++component)
        {
          // Now get the values of f(u)
          f_values[component] =
            this->rhs_function->value(fe_values.get_quadrature_points(),
                                      function_values,
                                      component);

        }



      // Now loop through test functions
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int component_i =
            fe_system.system_to_component_index(i).first;

          ADType R_i = 0;
          for (unsigned int q_index = 0;
               q_index < fe_values.n_quadrature_points;
               ++q_index)
            {
              // Integrate the cell term, i.e., \int_T_k < U' - f(U),
              // \psi > dt
              R_i += (function_partial_t
                        [component_i]
                        [q_index] - // Only want the partial derivative in time.
                                    // No new physics from making stochastic!
                      f_values[component_i][q_index]) *
                     fe_values.shape_value_component(i, q_index, component_i) *
                     fe_values.JxW(q_index);
            }

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            copy_data.cell_matrix(i, j) += R_i.fastAccessDx(j);

          copy_data.cell_rhs(i) -= R_i.val();
        }
    } //;
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    assemble_face_terms_for_residual_linearization(
      const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
      const unsigned int &face_index,
      const unsigned int &subface_index,
      const typename DoFHandler<dim + stochdim>::active_cell_iterator
                                             &cell_neighbor,
      const unsigned int                     &neighbor_face_index,
      const unsigned int                     &neighbor_subface_index,
      Auxiliary::ScratchData<dim + stochdim> &scratch_data,
      Auxiliary::CopyData                    &copy_data)
  {
    {
      if (face_index * neighbor_face_index != 0)
        return;
      // Evaluate <u_{n-1}^+ - u_{n-1}^-, v^+>
      // Need to grab the values from the two cells at the faces...
      FEInterfaceValues<dim + stochdim> &fe_iv =
        scratch_data.fe_interface_values;
      fe_iv.reinit(cell,
                   face_index,
                   subface_index,
                   cell_neighbor,
                   neighbor_face_index,
                   neighbor_subface_index);

      const auto &q_points = fe_iv.get_quadrature_points();

      copy_data.face_data.emplace_back();
      Auxiliary::CopyDataFace &copy_data_face = copy_data.face_data.back();

      // The total number of dofs at this interface (i.e., from both cells)
      const unsigned int n_dofs        = fe_iv.n_current_interface_dofs();
      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();
      //      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);
      copy_data_face.reinit(n_dofs);

      // We need to evaluate the jump in the solution at the interface using
      // the current guess for the solution

      // Need the AD equivalents for the DoF values at the interface
      std::vector<ADType> interface_dof_values(n_dofs);
      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          // Grab the actual value for this variable
          interface_dof_values[i] =
            this->solution(copy_data_face.joint_dof_indices[i]);
          // Let the AD system know we need partials wrt this
          interface_dof_values[i].diff(i, n_dofs);
        }

      // Now, we need at each quadrature point the jump in the function values
      std::vector<ADType> function_jump[n_components];
      for (unsigned int component = 0; component < n_components; ++component)
        function_jump[component].resize(q_points.size());

      for (unsigned int component = 0; component < n_components; ++component)
        fe_iv[this->fe_extractors[component]]
          .get_jump_in_function_values_from_local_dof_values(
            interface_dof_values, function_jump[component]);

      const std::vector<Tensor<1, dim + stochdim>> &normals =
        fe_iv.get_normal_vectors();

      // std::vector<double> residual_derivatives(n_dofs);
      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          // Get the pair of indices for this interace dof
          // Note that only 1 is actually valid, b/c DG ansatz

          const auto inter_local_indices =
            fe_iv.interface_dof_to_dof_indices(i);
          const unsigned int inter_local_index =
            inter_local_indices[0] == numbers::invalid_unsigned_int ?
              inter_local_indices[1] :
              inter_local_indices[0];

          const unsigned int component_i =
            fe_iv.get_fe().system_to_component_index(inter_local_index).first;
          // For each test function, we need to look at the residual
          ADType R_i = 0;
          for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
            {
              const double sign_factor = normals[qpoint][0] < 0 ? 1.0 : -1.0;
              R_i += fe_iv.shape_value(normals[qpoint][0] < 0,
                                       // Just using false doesn't work in the
                                       // case of N-irregular meshes, N > 0
                                       i,
                                       qpoint,
                                       component_i) *
                     sign_factor * function_jump[component_i][qpoint] *
                     fe_iv.JxW(qpoint);
            }

          // Integration of the residual complete, need partial wrt DoFs
          for (unsigned int j = 0; j < n_dofs; ++j)
            copy_data_face.cell_matrix(i, j) += R_i.fastAccessDx(j);
          // residual_derivatives[j] = R_i.fastAccessDx(j);

          // Now add the RHS contribution
          copy_data_face.cell_rhs(i) -= R_i.val();
        }
    } //;
  }

  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    assemble_boundary_terms_for_residual_linearization(
      const typename DoFHandler<dim + stochdim>::active_cell_iterator &cell,
      const unsigned int                                              &face_no,
      Auxiliary::ScratchData<dim + stochdim> &scratch_data,
      Auxiliary::CopyData                    &copy_data)
  {
    {
      const auto &face = cell->face(face_no);
      if (face->boundary_id() == 10)
        {
          scratch_data.fe_interface_values.reinit(cell, face_no);
          const auto &fe_face_values =
            scratch_data.fe_interface_values.get_fe_face_values(0);

          const auto        &q_points = fe_face_values.get_quadrature_points();
          const unsigned int n_facet_dofs =
            fe_face_values.get_fe().n_dofs_per_cell();

          copy_data.face_data.emplace_back();
          Auxiliary::CopyDataFace &copy_data_face = copy_data.face_data.back();
          copy_data_face.reinit(n_facet_dofs);
          copy_data_face.joint_dof_indices =
            scratch_data.fe_interface_values.get_interface_dof_indices();


          // Need the AD equivalents for the DoF values at the interface
          std::vector<ADType> interface_dof_values(n_facet_dofs);
          for (unsigned int i = 0; i < n_facet_dofs; ++i)
            {
              // Grab the actual value for this variable
              interface_dof_values[i] = this->solution(
                copy_data
                  .local_dof_indices[i]); // local_dof_indices is
                                          // stable in this case b/c of
                                          // how the meshworker operates...
              // Let the AD system know we need partials wrt this
              interface_dof_values[i].diff(i, n_facet_dofs);
            }

          std::vector<ADType> function_values[n_components];
          std::vector<double> ini_con_values[n_components];
          const auto &qpoint_vector = fe_face_values.get_quadrature_points();

          for (unsigned int component = 0; component < n_components;
               ++component)
            {
              function_values[component].resize(q_points.size());
              ini_con_values[component].resize(q_points.size());
            }

          for (unsigned int component = 0; component < n_components;
               ++component)
            {
              fe_face_values[this->fe_extractors[component]]
                .get_function_values_from_local_dof_values(
                  interface_dof_values, function_values[component]);

              initial_condition_function.value_list(qpoint_vector,
                                                    ini_con_values[component],
                                                    component);
            }

          for (unsigned int i = 0; i < n_facet_dofs; ++i)
            {
              const unsigned int component_i =
                fe_system.system_to_component_index(i).first;
              ADType R_i = 0;
              for (unsigned int qpoint = 0;
                   qpoint < fe_face_values.n_quadrature_points;
                   ++qpoint)
                R_i +=
                  (fe_face_values.shape_value(i, qpoint) *
                   (function_values[component_i][qpoint] -
                    /* Initial condition contrib*/ ini_con_values[component_i]
                                                                 [qpoint])) *
                  fe_face_values.JxW(qpoint);

              // Integration of the residual complete, need partial wrt DoFs
              for (unsigned int j = 0; j < n_facet_dofs; ++j)
                {

                  copy_data_face.cell_matrix(i, j) += R_i.fastAccessDx(j);
                }

              // Now add the RHS contribution

              copy_data_face.cell_rhs(i) -= R_i.val();
            }
        }
    } //;
  }
  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    execute_refinement()
  {
    std::cout << "DoFs before refinement: " << this->dof_handler.n_dofs()
              << std::endl;
    // Need to prepare a solution transfer so our next guess is easy
    SolutionTransfer<dim + stochdim, Vector<double>> soltrans(dof_handler);

    this->triangulation.prepare_coarsening_and_refinement();
    soltrans.prepare_for_coarsening_and_refinement(this->solution);

    // Apply refinement instructions
    this->triangulation.execute_coarsening_and_refinement();
    this->dof_handler.distribute_dofs(fe_system);

    std::cout << "DoFs after refinement: " << this->dof_handler.n_dofs()
              << std::endl;
    // Reinit old_solution
    this->old_solution.reinit(this->dof_handler.n_dofs());

    // Actually interpolate
    soltrans.interpolate(this->solution, this->old_solution);

    // Now, need to ensure the new guess satisfies the temporal boundary
    // condition
    this->update_initial_guess(this->old_solution);
    this->solution = old_solution;

    // Now we have to update the system sizes
    this->system_rhs.reinit(this->dof_handler.n_dofs());

    // Do same for the system matrix
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
  }
  template <int dim,
            int stochdim,
            class FEType,
            int n_components,
            int n_uncertain_parameters>
  void
  ODESolver<dim, stochdim, FEType, n_components, n_uncertain_parameters>::
    boost_grid(const unsigned int boost_by_level)
  {
    for (unsigned int i = 0; i < boost_by_level; ++i)
      {
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            cell->set_refine_flag();
            // cell->set_refine_flag(RefinementCase<dim + stochdim>::cut_axis(0));
          }

        this->execute_refinement();
      }
  }



} // namespace cG_and_dG

namespace TestingODESolver
{
  static constexpr size_t number_of_dependent_variables = 1;

  // The Observer is a class that stores the state of the ODE integration so we
  // can get not just the end result of the integration, but a solution over
  // time. That is, we have x(t) instead of just x(t_final).
  class Observer
  {
  public:
    void
    operator()(
      const std::array<double, number_of_dependent_variables> &current_x,
      const double current_time) noexcept
    {
      x.push_back(current_x[0]);
      time.push_back(current_time);
    }
    std::vector<double> x;
    std::vector<double> time;
  };

  void
  test_ode_integrator()
  {
    const double x_start_value = 0.0;
    const double x_end_value   = 2.0;
    // The Delta x is somewhat arbitrary. This is basically the size of the
    // rectangles in a Riemann sum. So if you remember doing Riemann sums to
    // estimate an integral by cutting up the area under the curve into a bunch
    // of rectangles, initial_delta_x_for_integration is the width of those
    // rectangles.
    //
    // This also determines how often we store the current solution in the
    // observer object below.
    const double initial_delta_x_for_integration = 0.1;

    // Tolerances means how accurate we want the solution. Relative tolerance is
    // "how many digits" while absolute tolerence is "we only care about the
    // solution if its value is larger than this".
    //
    // Absolute tolerance (safeguards against zeros is the solution):
    // if y_numerical - y_exact < absolute_tolerance:
    //   "things are fine, we don't care about tiny numbers"
    const double absolute_tolerance = 1.0e-8;
    // Relative tolerance:
    // (y_numerical - y_exact) / abs(y_numerical)
    const double relative_tolerance = 1.0e-8;

    // All these super long types are just Boost being.... "flexible" (I'd say
    // difficult, actually)

    using SolutionType    = dealii::Point<number_of_dependent_variables>;
    using ODESolutionType = std::array<double, number_of_dependent_variables>;

    using StateDopri5 =
      boost::numeric::odeint::runge_kutta_dopri5<ODESolutionType>;

    auto dopri5 =
      boost::numeric::odeint::make_controlled<StateDopri5>(absolute_tolerance,
                                                           relative_tolerance);

    // The observer object will store the result at specific times. Which times
    // can be controlled by choosing changing initial_delta_x_for_integration.
    Observer observer_fixed_step_size{};

    // The observer object will store the result at specific times. Which times
    // are specified in the times_to_observe_at std::vector<double>. The
    // integration range is from the first value to the last.
    Observer            observer_at_chosen_steps{};
    std::vector<double> times_to_observe_at{
      x_start_value, 0.23, 0.4444, 0.8888, 1.374, 1.843, x_end_value};

    // This is the initial condition and will be updated as we integrate.
    SolutionType x{{1.0}};
    // We want to solve:
    // dx / dt = x



    // Need to reset to initial condition before integrating again
    x[0] = 1.0;

    auto x_op = to_array(x);

    // Integrate while observing at specified times
    boost::numeric::odeint::integrate_times(
      dopri5,
      [](const ODESolutionType &current_value_of_x,
         ODESolutionType       &current_time_derivative_of_x,
         const double           current_time_t) noexcept {
        // Note we don't use the time explicitly!
        (void)current_time_t;
        // This computes the dx/dt
        current_time_derivative_of_x[0] = current_value_of_x[0];
      },
      x_op,
      times_to_observe_at.begin(),
      times_to_observe_at.end(),
      initial_delta_x_for_integration,
      std::ref(observer_at_chosen_steps));

    std::cout
      << "\n\nPrinting out solution obtained at explicitly chosen times.\n";
    for (size_t time_index = 0; time_index < observer_at_chosen_steps.x.size();
         ++time_index)
      {
        std::cout << observer_at_chosen_steps.time[time_index] << " "
                  << observer_at_chosen_steps.x[time_index] << "\n";
      }
  }
} // namespace TestingODESolver

int
main()
{
  try
    {
      using namespace cG_and_dG;

      std::cout << "Testing ODE integrator: " << std::endl;
      TestingODESolver::test_ode_integrator();
      std::cout << "Done testing ODE integrator!" << std::endl;

      // FOR 1-D Simulation (no stochasticity)
      // ODESolver<1, FE_DGQ<1, 1>> ODE_solver(4);
      // ODE_solver.run();

      // For 2-D Simulation (1-D stochastic space)
      {
        constexpr unsigned int dim      = 1;
        constexpr unsigned int stochdim = 1;



        auto stochastic_ini_con_0 = [=](const Point<stochdim> &) {
          return 1.0;
        };

        auto stochastic_ini_con_1 = [=](const Point<stochdim> &) { return 0; };

        auto stochastic_ini_con_2 = [=](const Point<stochdim> &p) {
          return 24; // + 1.0 * p[0];
        };

        LorenzRHS<stochdim, double, Sacado::Fad::DFad<double>> rhs_function;

        ODESolver<dim,
                  stochdim,
                  FE_DGQ<dim + stochdim, dim + stochdim>,
                  /* n components */ rhs_function.n_components,
                  /* n uncertain parameters*/ rhs_function.n_components>
          ODE_solver(2 /*order*/,
                     1.3, // 1.2063412712106940106 /* max time */,
                     &rhs_function /* rhs function */);
        ODE_solver.set_stochastic_domain(Point<stochdim>(0),
                                         Point<stochdim>(1));
        ODE_solver.add_stochastic_injector(
          0, stochastic_ini_con_0, UncertainParameterType::INITIAL_CONDITION);
        ODE_solver.add_stochastic_injector(
          1, stochastic_ini_con_1, UncertainParameterType::INITIAL_CONDITION);
        ODE_solver.add_stochastic_injector(
          2, stochastic_ini_con_2, UncertainParameterType::INITIAL_CONDITION);


        // Make stochastic injector for initial condition
        //         auto stochastic_ini_con = [=](const Point<stochdim> &p) {
        //           return 1.0; // + 1.0 * p[0];
        //         };

        // Generate the RHS function
        //         ExponentialRHS<stochdim, double, Sacado::Fad::DFad<double>>
        //           rhs_function;
        //         NonlinearScalarRHS<stochdim, double,
        //         Sacado::Fad::DFad<double>>
        //           rhs_function;
        //
        //         ODESolver<dim,
        //                  stochdim,
        //                  FE_DGQ<dim + stochdim, dim + stochdim>,
        //                   rhs_function.n_components,
        //                   rhs_function.n_components>
        //          ODE_solver(3,
        //                     0.75,
        //                     &rhs_function);
        //        ODE_solver.set_stochastic_domain(Point<stochdim>(0),
        //                                         Point<stochdim>(1));
        //        ODE_solver.add_stochastic_injector(
        //          0, stochastic_ini_con,
        //          UncertainParameterType::INITIAL_CONDITION);

        ODE_solver.run_newton();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
