//
// Created by harmon on 5/8/24.
//

#ifndef DECURION_RIGHTHANDSIDE_H
#define DECURION_RIGHTHANDSIDE_H



#include <deal.II/base/point.h>
#include <deal.II/differentiation/ad.h>

using namespace dealii;

template <int dim,
          int stochdim,
          int n_components,
          class Real   = double,
          class ADType = Sacado::Fad::DFad<double>>
class RightHandSide /*: public Function<dim + stochdim>*/
{
public:
  virtual Point<n_components>
  value(const Point<dim + stochdim> & /* p */,
        const Point<n_components> & /* u */) const = 0;

  virtual double
  value(const Point<dim + stochdim> & /* p */,
        const Point<n_components> & /* u */,
        const unsigned int & /* component */) const = 0;

  virtual std::vector<ADType>
  value(const std::vector<Point<dim + stochdim>> & /* p */,
        const std::array<std::vector<ADType>, n_components> & /* u */,
        const unsigned int & /* component */) const = 0;

  virtual std::vector<Real>
  value(const std::vector<Point<dim + stochdim>> & /* p */,
        const std::array<std::vector<Real>, n_components> & /* u */,
        const unsigned int & /* component */) const = 0;

  virtual void
  set_parameter_function(
    const unsigned int & /* param_index */,
    std::function<double(const Point<dim + stochdim> &)> /* param_f */) = 0;

  virtual FEValuesExtractors::Scalar
  get_component_extractor(const unsigned int &) = 0;

  virtual Point<n_components>
  adjoint_value(
    const Point<dim + stochdim> & /* p */,
    const Point<n_components> & /* adjoint solution \phi at p */,
    const Point<n_components> & /* forward solution u at p */) const = 0;

  virtual double
  adjoint_value(const Point<dim + stochdim> & /* p */,
                const Point<n_components> & /* adjoint solution \phi at p */,
                const Point<n_components> & /* forward solution u at p */,
                const unsigned int & /* component */) const = 0;

  virtual std::array<double, n_components>
  grad_multiplied(const Point<dim + stochdim>            &p,
                  const std::array<double, n_components> &phi,
                  const Point<n_components>              &u) const = 0;
};

template <int stochdim, class Real, class ADType>
class ExponentialRHS : public RightHandSide<1, stochdim, 1, Real, ADType>
{
public:
  static constexpr int dim          = 1;
  static constexpr int n_components = 1;


  ExponentialRHS()
  {
    parameters[0] = [](const Point<dim + stochdim> &) { return 2.0; };
  }

  virtual Point<n_components>
  value(const Point<dim + stochdim> &p,
        const Point<n_components>   &u) const override
  {
    const double c = this->parameters[0](p);

    Point<n_components> out(c * u[0]);

    return out;
  }

  virtual double
  value(const Point<dim + stochdim> &p,
        const Point<n_components>   &u,
        const unsigned int &) const override
  {
    return this->parameters[0](p) * u[0];
  }

  virtual std::vector<ADType>
  value(const std::vector<Point<dim + stochdim>>            &p,
        const std::array<std::vector<ADType>, n_components> &u,
        const unsigned int &) const override
  {
    std::vector<ADType> out(p.size());

    for (unsigned int i = 0; i < p.size(); ++i)
      out[i] = this->parameters[0](p[i]) * u[0][i];

    return out;
  }

  virtual std::vector<Real>
  value(const std::vector<Point<dim + stochdim>>          &p,
        const std::array<std::vector<Real>, n_components> &u,
        const unsigned int & /* component */) const override
  {
    std::vector<Real> out(p.size());

    for (unsigned int i = 0; i < p.size(); ++i)
      out[i] = this->parameters[0](p[i]) * u[0][i];

    return out;
  }

  virtual void
  set_parameter_function(
    const unsigned int                                  &param_index,
    std::function<double(const Point<dim + stochdim> &)> param_f) override
  {
    this->parameters[param_index] = std::move(param_f);
  }

  virtual FEValuesExtractors::Scalar
  get_component_extractor(const unsigned int &) override
  {
    return u;
  }

  virtual Point<n_components>
  adjoint_value(
    const Point<dim + stochdim> &p,
    const Point<n_components>   &phi,
    const Point<n_components> & /* forward solution u at p */) const override
  {
    Point<n_components> out(-this->parameters[0](p) * phi[0]);
    return out;
  }

  virtual double
  adjoint_value(const Point<dim + stochdim> &p,
                const Point<n_components>   &phi,
                const Point<n_components> &,
                const unsigned int &component) const override
  {
    return -this->parameters[0](p) * phi[0];
  }

  virtual std::array<double, n_components>
  grad_multiplied(const Point<dim + stochdim>            &p,
                  const std::array<double, n_components> &phi,
                  const Point<n_components>              &u) const override
  {
    return phi*this->parameters[0](p);
  }

private:
  // The (in this case) 1 parameter for the exponential problem: c
  // Depends on dim + stochdim, but probably just uses the stochdim portion...
  std::function<double(const Point<dim + stochdim> &)> parameters[n_components];

  // Since this is just a scalar problem, we have the one extractor
  FEValuesExtractors::Scalar u = FEValuesExtractors::Scalar(0);
};

template <int stochdim, class Real, class ADType>
class NonlinearScalarRHS : public RightHandSide<1, stochdim, 1, Real, ADType>
{
public:
  static constexpr int dim          = 1;
  static constexpr int n_components = 1;


  NonlinearScalarRHS()
  {
    parameters[0] = [](const Point<dim + stochdim> &) { return 1.0; };
  }

  virtual Point<n_components>
  value(const Point<dim + stochdim> &p,
        const Point<n_components>   &u) const override
  {
    const double c = this->parameters[0](p);

    Point<n_components> out(c * u[0] * u[0]);

    return out;
  }

  virtual double
  value(const Point<dim + stochdim> &p,
        const Point<n_components>   &u,
        const unsigned int &) const override
  {
    return this->parameters[0](p) * u[0] * u[0];
  }

  virtual std::vector<ADType>
  value(const std::vector<Point<dim + stochdim>>            &p,
        const std::array<std::vector<ADType>, n_components> &u,
        const unsigned int &) const override
  {
    std::vector<ADType> out(p.size());

    for (unsigned int i = 0; i < p.size(); ++i)
      out[i] = this->parameters[0](p[i]) * u[0][i] * u[0][i];

    return out;
  }

  virtual std::vector<Real>
  value(const std::vector<Point<dim + stochdim>>          &p,
        const std::array<std::vector<Real>, n_components> &u,
        const unsigned int & /* component */) const override
  {
    std::vector<Real> out(p.size());

    for (unsigned int i = 0; i < p.size(); ++i)
      out[i] = this->parameters[0](p[i]) * u[0][i] * u[0][i];

    return out;
  }

  virtual void
  set_parameter_function(
    const unsigned int                                  &param_index,
    std::function<double(const Point<dim + stochdim> &)> param_f) override
  {
    this->parameters[param_index] = std::move(param_f);
  }

  virtual FEValuesExtractors::Scalar
  get_component_extractor(const unsigned int &) override
  {
    return u;
  }

  virtual Point<n_components>
  adjoint_value(const Point<dim + stochdim> &p,
                const Point<n_components>   &phi,
                const Point<n_components>   &u) const override
  {
    Point<n_components> out(-this->parameters[0](p) * phi[0] *
                            u[0]); // Just copy pasted, check for correctness
    return out;
  }

  virtual double
  adjoint_value(const Point<dim + stochdim> &p,
                const Point<n_components>   &phi,
                const Point<n_components>   &u,
                const unsigned int          &component) const override
  {
    return -this->parameters[0](p) * phi[0] *
           u[0]; // Just copy pasted, check for correctness
  }

  virtual std::array<double, n_components>
  grad_multiplied(const Point<dim + stochdim>            &p,
                  const std::array<double, n_components> &phi,
                  const Point<n_components>              &u) const override
  {
    std::array<double, n_components> out;
      for (unsigned int i = 0; i < n_components; ++i)
        out[i] = this->parameters[0](p)*phi[i]*u[0];
      return out;
  }

private:
  // The (in this case) 1 parameter for the exponential problem: c
  // Depends on dim + stochdim, but probably just uses the stochdim portion...
  std::function<double(const Point<dim + stochdim> &)> parameters[n_components];

  // Since this is just a scalar problem, we have the one extractor
  FEValuesExtractors::Scalar u = FEValuesExtractors::Scalar(0);
};

template <int stochdim, class Real, class ADType>
class LorenzRHS : public RightHandSide<1, stochdim, 3, Real, ADType>
{
public:
  static constexpr int dim          = 1;
  static constexpr int n_components = 3;


  LorenzRHS()
  {
    parameters[0] /* sigma */ = [](const Point<dim + stochdim> &) {
      return 10.0;
    };
    parameters[1] /* r */ = [](const Point<dim + stochdim> &) { return 28.0; };
    parameters[2] /* b */ = [](const Point<dim + stochdim> &) {
      return 8.0 / 3.0;
    };
  }

  virtual Point<n_components>
  value(const Point<dim + stochdim> &p,
        const Point<n_components>   &u) const override
  {
    const auto sigma = this->parameters[0](p);
    const auto r     = this->parameters[1](p);
    const auto b     = this->parameters[2](p);

    Point<n_components> out = {sigma * (u[1] - u[0]),
                               r * u[0] - u[1] - u[0] * u[2],
                               u[0] * u[1] - b * u[2]};
    return out;
  }

  virtual double
  value(const Point<dim + stochdim> &p,
        const Point<n_components>   &u,
        const unsigned int          &component) const override
  {
    if (component == 0)
      return this->parameters[0](p) * (u[1] - u[0]);
    else if (component == 1)
      return this->parameters[1](p) * u[0] - u[1] - u[0] * u[2];
    else
      return u[0] * u[1] - this->parameters[2](p) * u[2];
  }

  virtual std::vector<ADType>
  value(const std::vector<Point<dim + stochdim>>            &p,
        const std::array<std::vector<ADType>, n_components> &u,
        const unsigned int &component) const override
  {
    std::vector<ADType> out(p.size());

    if (component == 0)
      for (unsigned int i = 0; i < p.size(); ++i)
        {
          const auto &cur_point = p[i];
          out[i] = this->parameters[0](cur_point) * (u[1][i] - u[0][i]);
        }
    else if (component == 1)
      for (unsigned int i = 0; i < p.size(); ++i)
        {
          const auto &cur_point = p[i];
          out[i] = this->parameters[1](cur_point) * u[0][i] - u[1][i] -
                   u[0][i] * u[2][i];
        }
    else
      for (unsigned int i = 0; i < p.size(); ++i)
        {
          const auto &cur_point = p[i];
          out[i] = u[0][i] * u[1][i] - this->parameters[2](cur_point) * u[2][i];
        }

    return out;
  }

  virtual std::vector<Real>
  value(const std::vector<Point<dim + stochdim>>          &p,
        const std::array<std::vector<Real>, n_components> &u,
        const unsigned int &component) const override
  {
    std::vector<Real> out(p.size());

    if (component == 0)
      for (unsigned int i = 0; i < p.size(); ++i)
        {
          const auto &cur_point = p[i];
          out[i] = this->parameters[0](cur_point) * (u[1][i] - u[0][i]);
        }
    else if (component == 1)
      for (unsigned int i = 0; i < p.size(); ++i)
        {
          const auto &cur_point = p[i];
          out[i] = this->parameters[1](cur_point) * u[0][i] - u[1][i] -
                   u[0][i] * u[2][i];
        }
    else
      for (unsigned int i = 0; i < p.size(); ++i)
        {
          const auto &cur_point = p[i];
          out[i] = u[0][i] * u[1][i] - this->parameters[2](cur_point) * u[2][i];
        }

    return out;
  }

  virtual void
  set_parameter_function(
    const unsigned int                                  &param_index,
    std::function<double(const Point<dim + stochdim> &)> param_f) override
  {
    this->parameters[param_index] = std::move(param_f);
  }

  virtual FEValuesExtractors::Scalar
  get_component_extractor(const unsigned int &i) override
  {
    switch (i)
      {
        case 0:
          return y_1;
        case 1:
          return y_2;
        case 2:
          return y_3;
        default:
          {
            assert(false && "Must specify a valid index i < 3 for LorenzRHS!");
            return y_1;
          }
      }
  }

  /**
   *
   * @param p
   * @param phi
   * @param u
   * @return
   */
  virtual Point<n_components>
  adjoint_value(const Point<dim + stochdim> &p,
                const Point<n_components>   &phi,
                const Point<n_components>
                  &u /* u at p at time (terminal_time - t)*/) const override
  {
    const auto sigma = this->parameters[0](p);
    const auto r     = this->parameters[1](p);
    const auto b     = this->parameters[2](p);

    const auto &x = u[0];
    const auto &y = u[1];
    const auto &z = u[2];

    const auto &phi0 = phi[0];
    const auto &phi1 = phi[1];
    const auto &phi2 = phi[2];

    Point<n_components> out = {-phi0 * sigma + phi1 * r - phi1 * z + phi2 * y,
                               phi0 * sigma - phi1 + phi2 * x,
                               -x * phi1 - b * phi2};
    return out;
  }

  virtual double
  adjoint_value(const Point<dim + stochdim> &p,
                const Point<n_components>   &phi,
                const Point<n_components>   &u,
                const unsigned int          &component) const override
  {
    switch (component)
      {
        case 0:
          return -phi[0] * this->parameters[0](p) +
                 phi[1] * this->parameters[1](p) - phi[1] * u[2] +
                 phi[2] * u[1];
          break;
        case 1:
          return phi[0] * this->parameters[0](p) - phi[1] + phi[2] * u[0];
          break;
        case 2:
        default:
          return -u[0] * phi[1] - this->parameters[2](p) * phi[2];
      }
  }

  virtual std::array<double, n_components>
  grad_multiplied(const Point<dim + stochdim>            &p,
                  const std::array<double, n_components> &phi,
                  const Point<n_components>              &u) const override
  {

    const auto sigma = this->parameters[0](p);
    const auto r     = this->parameters[1](p);
    const auto b     = this->parameters[2](p);

    const auto &x = u[0];
    const auto &y = u[1];
    const auto &z = u[2];

    const auto &phi0 = phi[0];
    const auto &phi1 = phi[1];
    const auto &phi2 = phi[2];

    std::array<double, n_components> out;
    out[0] = -sigma * phi0 + phi1 * (r - z) - phi2 * y;
    out[1] = sigma * phi0 - phi1 + phi2 * x;
    out[2] = -phi1 * x - phi2 * b;
    return out;
  }

private:
  // The (in this case) 3 parameters for the Lorenz system: \sigma, r, b
  // Depends on dim + stochdim, but probably just uses the stochdim portion...
  // (unless also depends on time, and potentially space)
  std::function<double(const Point<dim + stochdim> &)> parameters[n_components];

  // For this problem, we have 3 FEValuesExtractors
  FEValuesExtractors::Scalar y_1 = FEValuesExtractors::Scalar(0);
  FEValuesExtractors::Scalar y_2 = FEValuesExtractors::Scalar(1);
  FEValuesExtractors::Scalar y_3 = FEValuesExtractors::Scalar(2);
};

#endif // DECURION_RIGHTHANDSIDE_H
