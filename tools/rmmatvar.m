function rmmatvar(matfile, varname)
tmp = rmfield(load(matfile), varname);
save(matfile, '-struct', 'tmp');
end