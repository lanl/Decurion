//
// Created by harmon on 2/9/24.
//

#ifndef DECURION_RANDOMVARIABLES_H
#define DECURION_RANDOMVARIABLES_H

#include <boost/math/distributions/beta.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

#include <random>


namespace Stochastic
{
  template <class Real = double>
  struct RV
  {
    virtual Real
    evaluate_pdf(const Real &coord) const = 0;
    virtual Real
    evaluate_cdf(const Real &coord) const = 0;
    virtual Real
    sample() = 0;

    /**
     * Computes the expected value of $x$ over the region $[a,b]$
     * @param a
     * @param b
     * @return
     */
    virtual Real
    expected_value_x(const Real &lower, const Real &upper) const = 0;
  };

  template <class Real = double>
  struct UniformRV : public RV<Real>
  {
    UniformRV(const Real &a, const Real &b);

    Real
    evaluate_pdf(const Real &coord) const override;

    Real
    evaluate_cdf(const Real &coord) const override;

    Real
    sample() override;

    Real
    expected_value_x(const Real &lower, const Real &upper) const override;


    const Real a;
    const Real b;

  private:
    std::default_random_engine           generator;
    std::uniform_real_distribution<Real> uniformRealDistribution;
  };

  template <class Real>
  UniformRV<Real>::UniformRV(const Real &a, const Real &b)
    : a(a)
    , b(b)
    , uniformRealDistribution(a, b)
  {}
  template <class Real>
  Real
  UniformRV<Real>::evaluate_pdf(const Real &coord) const
  {
    if (coord > b || coord < a)
      return 0;
    else
      return 1.0 / (b - a);
  }
  template <class Real>
  Real
  UniformRV<Real>::evaluate_cdf(const Real &coord) const
  {
    if (coord < a)
      return 0;
    else if (coord > b)
      return 1;
    else
      return (coord - a) / (b - a);
  }
  template <class Real>
  Real
  UniformRV<Real>::sample()
  {
    return this->uniformRealDistribution(generator);
  }
  template <class Real>
  Real
  UniformRV<Real>::expected_value_x(const Real &lower, const Real &upper) const
  {
    return Real(0.5) * (upper * upper - lower * lower) / (b - a);
  }

  template <class Real = double>
  struct BetaRV : public RV<Real>
  {
    BetaRV(const Real &a, const Real &b);

    Real
    evaluate_pdf(const Real &coord) const override;

    Real
    evaluate_cdf(const Real &coord) const override;

    Real
    sample() override;

    Real
    expected_value_x(const Real &lower, const Real &upper) const;

    const Real a;
    const Real b;

    Real
    operator()(const Real &coord) const
    {
      return coord * this->evaluate_pdf(coord);
    }

    static constexpr Real lower_bound = 0.0;
    static constexpr Real upper_bound = 1.0;

  private:
    std::default_random_engine generator;
    // std::gamma_distribution<Real> gamma_distr_0, gamma_distr_1;
    boost::math::beta_distribution<Real>   beta_distribution_structure;
    boost::random::beta_distribution<Real> beta_distribution;
    boost::random::mt19937                 rng_gen;
  };
  template <class Real>
  BetaRV<Real>::BetaRV(const Real &a, const Real &b)
    : a(a)
    , b(b)
    , beta_distribution_structure(a, b)
    , beta_distribution(a, b)
  //    , gamma_distr_0(a, 1)
  //    , gamma_distr_1(1, b)
  {}
  template <class Real>
  Real
  BetaRV<Real>::evaluate_pdf(const Real &coord) const
  {
    return boost::math::pdf(beta_distribution_structure, coord);
  }
  template <class Real>
  Real
  BetaRV<Real>::evaluate_cdf(const Real &coord) const
  {
    return boost::math::cdf(beta_distribution_structure, coord);
  }
  template <class Real>
  Real
  BetaRV<Real>::sample()
  {
    return beta_distribution(rng_gen);
  }
  template <class Real>
  Real
  BetaRV<Real>::expected_value_x(const Real &lower, const Real &upper) const
  {
    return boost::math::quadrature::gauss<Real, 9>::integrate(*this,
                                                              lower,
                                                              upper);
  }

  template <class Real = double>
  struct GaussianRV : public RV<Real>
  {
    GaussianRV(/* Mean */ const Real &mu, /* Standard Deviation */ const Real &sigma);

    Real
    evaluate_pdf(const Real &coord) const override;

    Real
    evaluate_cdf(const Real &coord) const override;

    Real
    sample() override;

    Real
    expected_value_x(const Real &lower, const Real &upper) const;

    const Real mu;
    const Real sigma;

    Real
    operator()(const Real &coord) const
    {
      return coord * this->evaluate_pdf(coord);
    }

    static constexpr Real lower_bound = std::numeric_limits<Real>::min();
    static constexpr Real upper_bound = std::numeric_limits<Real>::max();

  private:
    std::default_random_engine generator;
    // std::gamma_distribution<Real> gamma_distr_0, gamma_distr_1;
    boost::math::normal_distribution<Real>   normal_distribution_structure;
    boost::random::normal_distribution<Real> normal_distribution;
    boost::random::mt19937                 rng_gen;
  };

  template <class Real>
  GaussianRV<Real>::GaussianRV(const Real &mu, const Real &sigma)
    : mu(mu)
    , sigma(sigma)
    , normal_distribution_structure(mu, sigma)
    , normal_distribution(mu, sigma)
  //    , gamma_distr_0(a, 1)
  //    , gamma_distr_1(1, b)
  {}

  template <class Real>
  Real
  GaussianRV<Real>::evaluate_pdf(const Real &coord) const
  {
    return boost::math::pdf(normal_distribution_structure, coord);
  }
  template <class Real>
  Real
  GaussianRV<Real>::evaluate_cdf(const Real &coord) const
  {
    return boost::math::cdf(normal_distribution_structure, coord);
  }
  template <class Real>
  Real
  GaussianRV<Real>::sample()
  {
    return normal_distribution(rng_gen);
  }
  template <class Real>
  Real
  GaussianRV<Real>::expected_value_x(const Real &lower, const Real &upper) const
  {
    return boost::math::quadrature::gauss<Real, 9>::integrate(*this,
                                                              lower,
                                                              upper);
  }
}

#endif // DECURION_RANDOMVARIABLES_H
