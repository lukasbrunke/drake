result = load('stewart_platform_36_binary/stewart_platform_result.txt');
analytical_result = load('stewart_platform_analytical_result.txt');
match = ones(40, 1);
match_error = zeros(40, 1);
for i = 1: 40
  [match_error(i), match(i)] = min(sum((result - bsxfun(@times, ones(size(result, 1), 1), analytical_result(i, :))).^2, 2));
end