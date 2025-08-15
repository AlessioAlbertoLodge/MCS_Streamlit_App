def load_emperical_degradation_parameters(chemistry, model="1"):
    if chemistry == 'LFP':
        if model == "1":  # A123 ANR26650M1
            a = 2.0916e-8
            b = -1.2179e-5
            c = 0.0018
            d = -1.7082e-6
            e = 0.0556
            f = 5.9808e6
            g = 0.6898
            h = -6.4647e3
            z =0.5
        elif model == "2":  # SONY US26650FT
            a = 2.9961e-8
            b = -1.7339e-5
            c = 0.0025
            d = -0.0124
            e = 3.8738
            f = 6.4726e8
            g = 1.4219
            h = -8.2191e3
            z =0.5

    elif chemistry == 'NMC':
        if model == "1":  # NMC 20 Ah
            a = 1.2787e-7
            b = -7.6845e-5
            c = 0.0116
            d = -0.0065
            e = -1.3745
            f = 1.1363e12
            g = 4.6996
            h = -1.0772e4
            z =0.5

        elif model == "2":  # SANYO UR18650 W
            a = 1.2918e-7
            b = -7.6878e-5
            c = 0.0114
            d = -6.7149e-3
            e = 2.3467
            f = 154.9601
            g = 0.6898
            h = -2.9467e3
            z =0.5

    empericial_ageing_parameters = [ a, b, c, d, e,f, g, h, z]
    return empericial_ageing_parameters
