//
// Created by harmon on 2/12/24.
//

#ifndef DECURION_QUADRATURE_H
#define DECURION_QUADRATURE_H

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <vector>

namespace quadrature
{
  template <int stochdim>
  void
  advance_stochastic_qpoint_index(
    dealii::Point<stochdim, int>       &index,
    const dealii::Point<stochdim, int> &n_quad_pts,
    int                                          trial_index = stochdim - 1)
  {
    ++index[trial_index];
    if (index[trial_index] >= n_quad_pts[trial_index] && trial_index > 0)
      {
        index[trial_index] = 0;
        advance_stochastic_qpoint_index<stochdim>(index, n_quad_pts, trial_index - 1);
      }
  }


  template <unsigned int NGL, class Real = double, bool Lobatto = false>
  class QGauss
  {
  public:
    template <class F>
    Real
    integrate(F &f, const Real &a, const Real &b) const
    {
      assert(false &&
             "Must use or provide a specialization for a specific NGL!");
    }
    Real
    integrate(Real values[], const Real &a, const Real &b) const
    {
      assert(false &&
             "Must use or provide a specialization for a specific NGL!");
    }
  };

  template <class Real>
  class QGauss<4, Real, /* Lobatto */ true>
  {
  public:
    template <class F>
    Real
    integrate(F &f, const Real &a, const Real &b) const
    {
      const Real average   = (a + b) / Real(2);
      const Real half_diff = (b - a) / Real(2);
      Real       out       = 0;
      for (unsigned int i = 0; i < 4; ++i)
        {
          const Real x_i = half_diff * abscissa[i] + average;
          out += f(x_i) * weights[i];
        }
      return out * half_diff;
    }
    unsigned int
    size() const
    {
      return 4;
    }
    const Real &
    get_weight(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 4 && "Index out of range!");
#endif
      return weights[index];
    }
    const Real &
    get_abscissa(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 4 && "Index out of range!");
#endif
      return abscissa[index];
    }

  private:
    const Real weights[4] = {1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0};

    const Real abscissa[4] = {-1.0,
                              -std::sqrt(5) * 0.2,
                              std::sqrt(5) * 0.2,
                              1.0};
  };

  template <class Real>
  class QGauss<6, Real, /* Lobatto */ true>
  {
  public:
    template <class F>
    Real
    integrate(F &f, const Real &a, const Real &b) const
    {
      const Real average   = (a + b) / Real(2);
      const Real half_diff = (b - a) / Real(2);
      Real       out       = 0;
      for (unsigned int i = 0; i < 6; ++i)
        {
          const Real x_i = half_diff * abscissa[i] + average;
          out += f(x_i) * weights[i];
        }
      return out * half_diff;
    }
    unsigned int
    size() const
    {
      return 6;
    }
    const Real &
    get_weight(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 6 && "Index out of range!");
#endif
      return weights[index];
    }
    const Real &
    get_abscissa(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 6 && "Index out of range!");
#endif
      return abscissa[index];
    }

  private:
    const Real weights[6] = {1.0 / 15.0,
                             (14.0 - sqrt(7.0)) / 30.0,
                             (14.0 + sqrt(7.0)) / 30.0,
                             (14.0 + sqrt(7.0)) / 30.0,
                             (14.0 - sqrt(7.0)) / 30.0,
                             1.0 / 15.0};

    const Real abscissa[6] = {
      -1.0,
      -std::sqrt(1.0 / 3.0 + 2.0 * std::sqrt(7.0) / 21.0),
      -std::sqrt(1.0 / 3.0 - 2.0 * std::sqrt(7.0) / 21.0),
      std::sqrt(1.0 / 3.0 - 2.0 * std::sqrt(7.0) / 21.0),
      std::sqrt(1.0 / 3.0 + 2.0 * std::sqrt(7.0) / 21.0),
      1.0};
  };

  template <class Real>
  class QGauss<2, Real>
  {
  public:
    template <class F>
    Real integrate(F &f, const Real& a, const Real& b) const
    {
      const Real average   = (a + b) / Real(2);
      const Real half_diff = (b - a) / Real(2);
      Real       out       = 0;
      for (unsigned int i = 0; i < 2; ++i)
        {
          const Real x_i = half_diff * abscissa[i] + average;
          out += f(x_i) * weights[i];
        }
      return out * half_diff;
    }
    unsigned int
    size() const
    {
      return 2;
    }
    const Real &
    get_weight(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 2 && "Index out of range!");
#endif
      return weights[index];
    }
    const Real &
    get_abscissa(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 2 && "Index out of range!");
#endif
      return abscissa[index];
    }

  private:
    const Real weights[2] = {1.0, 1.0};

    const Real abscissa[2] = {-0.5773502691896257, 0.5773502691896257};
  };

  template <class Real>
  class QGauss<1, Real>
  {
  public:
    template <class F>
    Real integrate(F &f, const Real& a, const Real& b) const
    {
      const Real average   = (a + b) / Real(2);
      const Real half_diff = (b - a) / Real(2);
      Real       out       = 0;
      for (unsigned int i = 0; i < 1; ++i)
        {
          const Real x_i = half_diff * abscissa[i] + average;
          out += f(x_i) * weights[i];
        }
      return out * half_diff;
    }
    unsigned int
    size() const
    {
      return 1;
    }
    const Real &
    get_weight(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 1 && "Index out of range!");
#endif
      return weights[index];
    }
    const Real &
    get_abscissa(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 1 && "Index out of range!");
#endif
      return abscissa[index];
    }

  private:
    const Real weights[1] = {2.0};

    const Real abscissa[1] = {0.00000000001};
  };

  template <class Real>
  class QGauss<4, Real>
  {
  public:
    template <class F>
    Real
    integrate(F &f, const Real &a, const Real &b) const
    {
      const Real average   = (a + b) / Real(2);
      const Real half_diff = (b - a) / Real(2);
      Real       out       = 0;
      for (unsigned int i = 0; i < 4; ++i)
        {
          const Real x_i = half_diff * abscissa[i] + average;
          out += f(x_i) * weights[i];
        }
      return out * half_diff;
    }
    unsigned int
    size() const
    {
      return 4;
    }
    const Real &
    get_weight(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 4 && "Index out of range!");
#endif
      return weights[index];
    }
    const Real &
    get_abscissa(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 4 && "Index out of range!");
#endif
      return abscissa[index];
    }

  private:
    const Real weights[4] = {0.3478548451374538,
                             0.6521451548625461,
                             0.6521451548625461,
                             0.3478548451374538};

    const Real abscissa[4] = {-0.8611363115940526,
                              -0.3399810435848563,
                              0.3399810435848563,
                              0.8611363115940526};
  };

  template <class Real>
  class QGauss<9, Real>
  {
  public:
    template <class F>
    Real
    integrate(F &f, const Real &a, const Real &b) const
    {
      const Real average   = (a + b) / Real(2);
      const Real half_diff = (b - a) / Real(2);
      Real       out       = 0;
      for (unsigned int i = 0; i < 9; ++i)
        {
          const Real x_i = half_diff * abscissa[i] + average;
          out += f(x_i) * weights[i];
        }
      return out * half_diff;
    }
    unsigned int
    size() const
    {
      return 9;
    }
    const Real &
    get_weight(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 9 && "Index out of range!");
#endif
      return weights[index];
    }
    const Real &
    get_abscissa(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 9 && "Index out of range!");
#endif
      return abscissa[index];
    }

  private:
    const Real weights[9] = {
      0.0812743883615744,
      0.1806481606948574,
      0.2606106964029354,
      0.3123470770400029,
      0.3302393550012598,
      0.3123470770400029,
      0.2606106964029354,
      0.1806481606948574,
      0.0812743883615744,
    };

    const Real abscissa[9] = {-0.9681602395076261,
                              -0.8360311073266358,
                              -0.6133714327005904,
                              -0.3242534234038089,
                              0.0000000000000000,
                              0.3242534234038089,
                              0.6133714327005904,
                              0.8360311073266358,
                              0.9681602395076261};
  };

  template <class Real>
  class QGauss<20, Real>
  {
  public:
    template <class F>
    Real
    integrate(F &f, const Real &a, const Real &b) const
    {
      const Real average   = (a + b) / Real(2);
      const Real half_diff = (b - a) / Real(2);
      Real       out       = 0;
      for (unsigned int i = 0; i < 20; ++i)
        {
          const Real x_i = half_diff * abscissa[i] + average;
          out += f(x_i) * weights[i];
        }
      return out * half_diff;
    }
    unsigned int
    size() const
    {
      return 20;
    }
    const Real &
    get_weight(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 20 && "Index out of range!");
#endif
      return weights[index];
    }
    const Real &
    get_abscissa(const unsigned int &index) const
    {
#ifdef DEBUG
      assert(index < 20 && "Index out of range!");
#endif
      return abscissa[index];
    }

  private:
    const Real weights[20] = {0.0176140071391521, 0.0406014298003869,
                              0.0626720483341091, 0.0832767415767048,
                              0.1019301198172404, 0.1181945319615184,
                              0.1316886384491766, 0.1420961093183820,
                              0.1491729864726037, 0.1527533871307258,
                              0.1527533871307258, 0.1491729864726037,
                              0.1420961093183820, 0.1316886384491766,
                              0.1181945319615184, 0.1019301198172404,
                              0.0832767415767048, 0.0626720483341091,
                              0.0406014298003869, 0.0176140071391521};

    const Real abscissa[20] = {-0.9931285991850949, -0.9639719272779138,
                               -0.9122344282513259, -0.8391169718222188,
                               -0.7463319064601508, -0.6360536807265150,
                               -0.5108670019508271, -0.3737060887154195,
                               -0.2277858511416451, -0.0765265211334973,
                               0.0765265211334973,  0.2277858511416451,
                               0.3737060887154195,  0.5108670019508271,
                               0.6360536807265150,  0.7463319064601508,
                               0.8391169718222188,  0.9122344282513259,
                               0.9639719272779138,  0.9931285991850949};
  };

} // namespace quadrature

namespace StochasticIntegration
{
  template <int stochdim, class ParentQuadratureRule>
  class TensorProductRule
  {
  public:
    TensorProductRule() = default;
    TensorProductRule(
      const dealii::Point<stochdim>               &lower,
      const dealii::Point<stochdim>               &upper,
      const dealii::Point<stochdim, int> &n_subdivisions)
    {
      // Subdivide the rectangular space and generate a tensor product rule
      const unsigned int n_quad_pts = quad_rule.size();
      for (unsigned int chosen_dim = 0; chosen_dim < stochdim; ++chosen_dim)
        {
          const double       &a     = lower[chosen_dim];
          const double       &b     = upper[chosen_dim];
          const unsigned int &n_sub = n_subdivisions[chosen_dim];

          const double interval_size      = (b - a) / double(n_sub);
          const double half_interval_size = interval_size * 0.5;

          for (unsigned int i = 0; i < n_sub; ++i)
            {
              const double average =
                (2.0 * a + i * interval_size + (i + 1) * interval_size) * 0.5;
              for (unsigned int j = 0; j < n_quad_pts; ++j)
                {
                  one_d_abscissa[chosen_dim].push_back(
                    half_interval_size * quad_rule.get_abscissa(j) + average);

                  one_d_weights[chosen_dim].push_back(quad_rule.get_weight(j) *
                                                      half_interval_size);
                }
            }
          one_d_n_quad_pts[chosen_dim] = one_d_abscissa[chosen_dim].size();
        }
      // We have now all the 1-D rules
      // Need to merge them in the standard tensor product fashion
      unsigned int n_total_pts = 1;
      for (unsigned int i = 0; i < stochdim; ++i)
        n_total_pts *= n_subdivisions[i]*n_quad_pts;

      abscissa = std::vector<dealii::Point<stochdim>>(n_total_pts);
      weights  = std::vector<double>(n_total_pts, 1.0);

      dealii::Point<stochdim, int> index;
      unsigned int n_added_pts = 0;
      while(n_added_pts < n_total_pts)
        {
          for (unsigned int chosen_dim = 0; chosen_dim < stochdim; ++chosen_dim)
            {
              abscissa[n_added_pts][chosen_dim] = one_d_abscissa[chosen_dim][index[chosen_dim]];
              weights[n_added_pts] *= one_d_weights[chosen_dim][index[chosen_dim]];
            }
          // increment the quad pt index
          quadrature::advance_stochastic_qpoint_index<stochdim>(index, one_d_n_quad_pts);
          ++n_added_pts;
        }
    }

    std::vector<dealii::Point<stochdim>> &
    get_abscissas()
    {
      return this->abscissa;
    }

    const std::vector<dealii::Point<stochdim>> &
    get_abscissas() const
    {
      return this->abscissa;
    }

    dealii::Point<stochdim>& get_abscissa(const unsigned int& index)
    {
      return this->abscissa[index];
    }

    const dealii::Point<stochdim>& get_abscissa(const unsigned int& index) const
    {
      return this->abscissa[index];
    }

    double& get_weight(const unsigned int& index)
    {
      return this->weights[index];
    }

    const double& get_weight(const unsigned int& index) const
    {
      return this->weights[index];
    }

    std::vector<double> &
    get_weights()
    {
      return weights;
    }

    const std::vector<double> &
    get_weights() const
    {
      return weights;
    }

    double
    integrate(const std::vector<double> &function_values) const
    {
      double out = 0.0;
      for (unsigned int i = 0; i < this->weights.size(); ++i)
        out += weights[i] * function_values[i];

      return out;
    }

    unsigned int size() const
    {
      return this->weights.size();
    }

  private:
    // The individual subrules
    std::vector<double> one_d_abscissa[stochdim], one_d_weights[stochdim];
    dealii::Point<stochdim, int> one_d_n_quad_pts;

    // The final tensor product rule
    std::vector<dealii::Point<stochdim>> abscissa;
    std::vector<double>                  weights;

    // Parent quadrature rule
    ParentQuadratureRule quad_rule;
  };
} // namespace StochasticIntegration

#endif // DECURION_QUADRATURE_H
