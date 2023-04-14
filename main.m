N = 256;

[X,Y] = meshgrid( (0:N-1)/N*(2*pi) );

%Use capital letters for grid
omega = cos(2*X) + sin(2*Y) + 0.5*sin(X+Y-7);

forcing = 4*sin(4*Y);
nu = 1/40;


o = zeros(N,N,M/every);

T = 10*16;
M = 1024*16;
every = 16;

a = 1;
vidObj = VideoWriter('NS.avi');
open(vidObj);
for steps = 1:M
  steps
  
  if( mod(steps-1, every) == 0 )
    imagesc(omega);
    set( gca, 'ydir', 'normal');
    colormap bluewhitered
    axis square
    colorbar
    caxis([-5 5]);
    drawnow

    % Write each frame to the file.
    currFrame = getframe(gcf);
    writeVideo(vidObj,currFrame);
    o(:,:,a) = omega;
    a = a+1;
  end
  
  num_steps = 1;
  omega = integrate_NS( omega, T/M, num_steps, nu, forcing );
end 
% Close the file.
close(vidObj);

h = T/M*every;
save('traj', 'o', 'h', 'forcing', 'nu');

%% Newton

%initial guess
omega = 10*(cos(2*X) + cos(Y - 1));
T = 2;
x = [ reshape(omega, [N*N,1]); T];

h = 1e-6;
M = 128;
nu = 1/40;

F = @(x) [(reshape( integrate_NS( reshape(x(1:N*N), [N,N]), x(end), M, nu, forcing ), [N*N,1]) - x(1:N*N))/x(end); 0 ];
for i = 1:128
  %need to redefine J since it depends on s
  f = F(x);
  [norm(f), x(end)]
  %J = @(s) (F(x+h*s) - F(x-h*s))/(2*h);
  J = @(s) Jacobian(x,s,h,F,N);
  
  hookstep = 0.1;
  step = gmres(J, f);
  x = x - hookstep*step;
  
  omega = reshape(x(1:N*N), [N,N]);
  imagesc(omega);
  colorbar();
  axis square
  
  drawnow
end

%% Animate result
T = x(end);
omega = reshape(x(1:N*N), [N,N]);
for steps = 1:M
  imagesc(omega);
  colormap bluewhitered
  axis square
  drawnow
  colorbar
  
  num_steps = 1;
  omega = integrate_NS( omega, T/M, num_steps, nu, forcing );
end

function b = Jacobian(x,s,h,F,N)
  b = (F(x+h*s) - F(x-h*s))/(2*h);
  omega = reshape( x(1:N*N), [N,N] );
  b(end) = dot( reshape(advection(omega),[N*N,1]), s(1:N*N) );
end

function advec = advection(omega)
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
  
  %dealias
  advec = fft2(advec);
  N = size(advec,1);
  advec( k_sq > (N/4)^2 ) = 0;
  advec = real(ifft2(advec));
end