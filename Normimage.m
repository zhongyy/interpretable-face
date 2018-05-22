function [rer] = Normimage( re )
    re = re(:, :, [3, 2, 1]); % convert from RGB to BGR
    re = permute(re, [2, 1, 3]); % permute width and height
    re=re-min(re(:));
    re=re/max(re(:));
    rer=round(re*255.0);
%     rer=imresize(rer,[250 250]);
end

