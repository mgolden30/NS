function omega = integrate_NS( omega, T, num_steps, nu, forcing )
  %{
  An integrator of the Navier-Stokes equations with periodic boundary
  conditions
  %}

  assert( size(omega,1) == size(omega,2) );
  
  N = size(omega,1);
  dt = T/num_steps;
 
  k = 0:N-1;
  k( k> N/2 ) = k( k> N/2 ) - N;
  
  k_sq = k.^2 + k'.^2;
  k_sq(1,1) = 1; %Otherwise we divide by zero
    
  for step = 1:num_steps
    %Create a velocity function
    rhs = @(omega) advection(omega, k, k_sq) + nu*laplacian(omega,k_sq) + forcing;
    
    
    %Let's do explcit RK6
    k1 = dt * rhs( omega );
    k2 = dt * rhs( omega + k1/3 );
    k3 = dt * rhs( omega         + k2*2/3 );
    k4 = dt * rhs( omega + k1/12 + k2/3   - k3/12 );
    k5 = dt * rhs( omega - k1/16 + 9/8*k2 - 3/16*k3 - 3/8*k4 );
    k6 = dt * rhs( omega         + 9/8*k2 - 3/8*k3 -3/4*k4 + k5/2 );
    k7 = dt * rhs( omega + k1*9/44 - 9/11*k2 + 63/44*k3 + 18/11*k4 - 16/11*k6 );
    
    omega = omega + 11/120*(k1 +k7) + 27/40*(k3+k4) - 4/15*(k5+k6);
    
    %Or RK4
    %{
    k1 = dt*rhs(omega);
    k2 = dt*rhs(omega + k1/2);
    k3 = dt*rhs(omega + k2/2);
    k4 = dt*rhs(omega + k3);
    
    omega = omega + (k1+2*k2+2*k3+k4);
    %}
    
    
    %dealias
    omega_fft = fft2(omega);
    omega_fft( k_sq >= (N/3)^2 ) = 0;
    omega = real(ifft2(omega_fft));
  end
  
  
end

function advec = advection(omega, k, k_sq)
  omega_fft = fft2( omega );

  %Calculate gradient
  omega_x = real( ifft2( 1i*    omega_fft.*k ) );
  omega_y = real( ifft2( 1i*k'.*omega_fft    ) );

  %Calculate velocity assuming mean flow is zero
  u = real( ifft2( 1i*k'.*omega_fft   ./k_sq    ) );
  v = real( ifft2(-1i*    omega_fft.*k./k_sq    ) );
  
  advec = -(u.*omega_x + v.*omega_y);
end

function lap_omega = laplacian( omega, k_sq)
  omega_fft = fft2( omega );

  temp = -omega_fft.*k_sq;
  temp(1,1) = 0; %no mean vorticity
  
  lap_omega = real( ifft2(temp) );
end