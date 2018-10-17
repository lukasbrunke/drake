result = load('stewart_platform_36_binary/stewart_platform_result.txt');
analytical_result = load('stewart_platform_analytical_result.txt');
% match(i) stores the best match in result that matches
% analytical_result(i, :)
match = ones(40, 1);
match_error = zeros(40, 1);
for i = 1: 40
  [match_error(i), match(i)] = min(sum((result - bsxfun(@times, ones(size(result, 1), 1), analytical_result(i, :))).^2, 2));
end

% not_match(i) stores the best match in analytical_result that matches
% result(i, :)
not_match = ones(size(result, 1), 1);
not_match_error = ones(size(result, 1), 1);
for i = 1 : size(result, 1)
    [not_match_error(i), not_match(i)] = min(sum(((bsxfun(@times, result(i,:), ones(40, 1)) - analytical_result)).^2, 2));
end