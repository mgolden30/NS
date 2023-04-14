%{
Playground for Crank-Nicolson
%}

dt = 0.1; % timestep
N  = 128;  % number of points in space

[X,Y] = meshgrid( (0:N-1)/N*(2*pi) );

omega = sin(X).*sin(Y) + sin(2*X).*sin(2*Y) + 0.1*cos(X-1).*sin(Y+1);
%omega = sin(2*X).*cos(Y) + sin(3*Y-1).*cos(X + 3);

forcing = 0*sin(X).*sin(Y);
nu = 1e-3;

vel = @(omega) rhs(omega, nu,forcing);


vidObj = VideoWriter('peaks.avi');
open(vidObj);

omega_next = omega + dt*vel(omega);
for i = 1:1024
  [i, diff]
  for j = 1:1024
  v = vel(omega);
  phi = @(omega) omega + (v + vel(omega_next))/2*dt;
  alpha = 1/8;
  
  omega2 = (1-alpha)*omega_next + alpha*phi(omega);
  
  diff = log10(norm(omega2 - omega_next));
  omega_next = omega2;
  if( diff < -14 )
     %converged
     continue;
  end
  
  end
  
  imagesc(X(1,:), X(1,:), omega_next);
  axis square
  set(gca, 'color', 'w');
  set(gca, 'ydir', 'normal');
  
  xticks([]);
  yticks([]);
  caxis([-1 1]);
  drawnow
  % Write each frame to the file.
       currFrame = getframe(gcf);
       writeVideo(vidObj,currFrame);
  omega = omega_next;
end
% Close the file.
close(vidObj);

function v = rhs( omega, nu, forcing )
  %{
  An integrator of the Navier-Stokes equations with periodic boundary
  conditions
  %}

  assert( size(omega,1) == size(omega,2) );
  N = size(omega,1);
  
  k = 0:N-1;
  k( k> N/2 ) = k( k> N/2 ) - N;
  
  k_sq = k.^2 + k'.^2;
  k_sq(1,1) = 1; %Otherwise we divide by zero
  
  omega_fft = fft2( omega );

  %Calculate gradient
  omega_x = real( ifft2( 1i*    omega_fft.*k ) );
  omega_y = real( ifft2( 1i*k'.*omega_fft    ) );

  %Calculate velocity assuming mean flow is zero
  u = real( ifft2( 1i*k'.*omega_fft   ./k_sq    ) );
  v = real( ifft2(-1i*    omega_fft.*k./k_sq    ) );
  
  advec = -(u.*omega_x + v.*omega_y);
  
  v = advec - nu*real( ifft2( k_sq.*omega_fft) ) + forcing;
end