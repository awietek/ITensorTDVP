#ifndef TDVP_KRYLOV_H_
#define TDVP_KRYLOV_H_

#include <chrono>
#include <lila/all.h>
#include <itensor/all.h>

using namespace itensor;

// #define DURATION(label, t1, t2) std::cout<<label<<" "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<"\n"
// #define CLK(t) t=std::chrono::high_resolution_clock::now()

inline void krylov(MPS& psi, const MPO& H, Real tau, const Args& args = Args::global())
{
  double tol = args.getReal("Cutoff", 1e-8);
  const double beta_tol = 1e-7;
  int max_iter = args.getInt("MaxIter", 10);
  psi.orthogonalize();
  auto arithmetic_args = Args("Cutoff=", 1e-12);

  // Initialize Lanczos vectors
  MPS v1 = psi;
  MPS v0;
  MPS w;

  Real nrm = norm(v1);
  v1.normalize();
  v1.position(1);

  std::vector<MPS> lanczos_vectors({v1});
  lila::Matrix<double> bigTmat(max_iter + 2, max_iter + 2);

  double beta = 0;
  for (int iter=0; iter < max_iter; ++iter)
    {	 
      // auto CLK(t1);
      // auto CLK(t2);
 
      int tmat_size=iter+1;
      
      // // Vector orthogonalization
      // CLK(t1);

      // auto CLK(tt1);
      double avnorm = sqrt(inner(H, v1, H, v1));
      // auto CLK(tt2);
      // DURATION("C1 ", tt1, tt2);

      // CLK(tt1);
      double alpha = inner(v1, H, v1);
      // CLK(tt2);
      // DURATION("C2 ", tt1, tt2);

      // CLK(tt1);
      double beta_prev = (iter > 0) ? bigTmat(iter-1, iter) : 0;
      // CLK(tt2);
      // DURATION("C3 ", tt1, tt2);

      beta = sqrt(avnorm*avnorm - alpha*alpha - beta_prev*beta_prev);
      // CLK(t2);
      // DURATION("COEFF ", t1, t2);

      bigTmat(iter, iter) = alpha;
      bigTmat(iter+1, iter) = beta;
      bigTmat(iter, iter+1) = beta;

     

      // check for Lanczos sequence exhaustion
      if (std::abs(beta) < beta_tol) 
	{	   
	  // printf("exhausted %d\n", tmat_size);
	  // Assemble the time evolved state
	  auto tmat = bigTmat;
	  tmat.resize(tmat_size, tmat_size);
	  auto tmat_exp = lila::ExpM(tmat, -tau);
	  auto linear_comb = tmat_exp.col(0);

	  // assemble the Lanczos Vectors
	  assert(lanczos_vectors.size() == linear_comb.size());
	  
	  // psi = nrm*linear_comb(0)*lanczos_vectors[0];
	  // for (int i=1; i<(int)lanczos_vectors.size(); ++i)
	  //   psi.plusEq(nrm*linear_comb(i)*lanczos_vectors[i], arithmetic_args);

	  for (int i=0; i<(int)lanczos_vectors.size(); ++i)
	    lanczos_vectors[i] *= nrm * linear_comb(i);
	  psi = sum(lanczos_vectors, arithmetic_args);

	  break;
	}
      

      // Convergence check
      // CLK(t1);
      if (iter > 0)
	{
	  // Prepare extended T-matrix for exponentiation
	  // Print(beta);
	  // LilaPrint(bigTmat);

	  auto tmat_ext = bigTmat;
	  int tmat_ext_size = tmat_size + 2;
	  tmat_ext.resize(tmat_ext_size, tmat_ext_size);
	  tmat_ext(tmat_size-1, tmat_size) = 0.;
	  tmat_ext(tmat_size+1, tmat_size) = 1.;
	  // LilaPrint(tmat_ext);

	  // Exponentiate extended T-matrix
	  auto tmat_ext_exp = lila::ExpM(tmat_ext, -tau);

	  double phi1 = std::abs( nrm*tmat_ext_exp(tmat_size, 0) );
	  double phi2 = std::abs( nrm*tmat_ext_exp(tmat_size + 1, 0) * avnorm );
	  // printf("phi1: %g, phi2: %g\n", phi1, phi2);
	  double error;      
	  if (phi1 > 10*phi2) error = phi2;
	  else if (phi1 > phi2) error = (phi1*phi2)/(phi1-phi2);
	  else error = phi1;
	  if ((error < tol) || (iter == max_iter-1))
	    {
	      if (iter == max_iter-1) 
		printf("warning: lanczosTevol not converged in %d steps\n", max_iter);
	      // else
	      //   printf("converged in %d steps\n", iter);


	      // // Print inner product of lanczos vectors
	      // for (int i=0; i<(int)lanczos_vectors.size(); ++i)
	      // 	{
	      // 	  // Print(maxLinkDim(lanczos_vectors[i]));
	      // 	  for (int j=0; j<(int)lanczos_vectors.size(); ++j)
	      // 	    std::cout << inner(lanczos_vectors[i], lanczos_vectors[j]) << " ";
	      // 	  std::cout << "\n";
	      // 	}

		
	      // Assemble the time evolved state
	      auto tmat = bigTmat;
	      tmat.resize(tmat_size, tmat_size);
	      auto tmat_exp = lila::ExpM(tmat, -tau);
	      auto linear_comb = tmat_exp.col(0);

	      // assemble the Lanczos Vectors
	      assert(lanczos_vectors.size() == linear_comb.size());

	      for (int i=0; i<(int)lanczos_vectors.size(); ++i)
		lanczos_vectors[i] *= nrm * linear_comb(i);
	      psi = sum(lanczos_vectors, arithmetic_args);


	      // printf("lanczos iters: %d\n", iter);
	      break;
	    }
	}

	// CLK(t2);
	// DURATION("CONV", t1, t2);


	////////////////////////////////////////////
	// Create next Lanczos vector

	// Matrix-vector multiplication
	// CLK(t1);
	w = applyMPO(H, v1, arithmetic_args);
	w.replaceTags("1", "0");
	// CLK(t2);
	// double avnorm2 = norm(w);
	// DURATION("MVM ", t1, t2);


	// Orthogonalize against previous vectors
	// CLK(t1);
	auto v1t = -alpha * v1;
	w.plusEq(v1t, arithmetic_args);

	if (iter > 0)
	  {
	    // w -= beta * v0;
	    v1t = -beta_prev * v0;
	    w.plusEq(v1t);
	  }
	v0 = v1;

	// double alpha2 = inner(w, v1);
	// double beta2 = norm(w);
	// CLK(t2);
	// DURATION("ORTHO ", t1, t2);

	// update next lanczos vector
	v1 = w;
	v1 /= beta;
	lanczos_vectors.push_back(v1);


	// Print(avnorm);
	// Print(avnorm2);
	// Print(alpha);
	// Print(alpha2);
	// Print(beta);
	// Print(beta2);

      }  // Lanczos iteratrions

  psi.normalize();
}

#endif
