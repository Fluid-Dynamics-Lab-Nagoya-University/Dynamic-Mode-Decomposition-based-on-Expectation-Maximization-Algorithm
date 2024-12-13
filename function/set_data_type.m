function X = set_data_type(X,opt_data_type)
    if opt_data_type.flag_processing_unit_type == 1
        if opt_data_type.flag_precision_type == 1
            X = single(gpuArray(X)) ;
        elseif opt_data_type.flag_precision_type == 2
            X = double(gpuArray(X)) ;
        end
    elseif opt_data_type.flag_processing_unit_type == 0
        if opt_data_type.flag_precision_type == 1
            X = single(gather(X)) ;
        elseif opt_data_type.flag_precision_type == 2
            X = double(gather(X)) ;
        end
    end
end