Start experiment try it out_n=100000_queries_n=100
rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 1 100000 k=30 ef=50 { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, stitch_mode: RandomNegToRandomPosAndBack } 34681070 10122.47 0.636 1370.130 0.943  
 1 100000 k=30 ef=50 { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, stitch_mode: RandomSubsetOfSubset } 460245146 166108.77 0.297 1012.660 0.666  
 1 100000 k=30 ef=50 { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, stitch_mode: BestXofRandomXTimesX } 11731922 2124.815 0.061 559.530 0.375  
 1 100000 k=30 ef=50 { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, stitch_mode: DontStarveXXSearch } 33401271 8641.363 0.583 1456.140 0.959

RandomSubsetOfSubset just takes too long

Start experiment try it out_n=30000_queries_n=100
rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 2 30000 k=30 ef=50 StitchingParams { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack } 9370907 2484.8535 0.710 1138.490 0.771  
 2 30000 k=30 ef=50 StitchingParams { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 9001535 2163.0542 0.642 1079.675 0.724

# params, with recall ca. 0.8 for 4 different models:

Start experiment try it out_n=30000_queries_n=100
rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 1 30000 dist=Dot k=30 ef=100 start_candidates=1 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 6488720 1619.7968 0.798 684.430 0.304  
 1 30000 dist=Dot k=30 ef=100 start_candidates=1 RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot } 6043928 1411.5428 0.794 851.000 0.392  
 1 30000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 256, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 4, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack } 8039595 2014.0679 0.791 1062.670 0.836  
 1 30000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 256, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 4, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 7925060 1766.496 0.792 1103.270 0.818

# stitching with 160k

Start experiment try it out_n=160000_queries_n=100
rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 1 160000 dist=Dot k=30 ef=100 start_candidates=1 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 40351556 19572.145 0.672 778.560 0.478  
 1 160000 dist=Dot k=30 ef=100 start_candidates=1 RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot } 32782910 8456.669 0.599 1126.340 0.577  
 1 160000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 256, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 10, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack } 47195121 13904.933 0.664 1330.320 0.923  
 1 160000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 256, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 10, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 47648609 11166.344 0.639 1387.370 1.002

# all about 0.74 recall:

rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 1 50000 dist=Dot k=30 ef=100 start_candidates=1 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 11324875 3720.5044 0.745 698.180 0.410  
 1 50000 dist=Dot k=30 ef=100 start_candidates=1 RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot } 10206200 2647.59 0.740 896.680 0.443  
 1 50000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack } 11348170 3127.3928 0.752 1185.370 0.857  
 1 50000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 11385147 2976.142 0.740 1251.780 0.891

# ensemble is superior

    rep    n        search_params                              params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean

1 50000 dist=Dot k=30 ef=100 start_candidates=1 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 11506835 4235.212 0.761 705.770 0.419  
 1 50000 dist=Dot k=30 ef=100 start_candidates=1 RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot } 10153597 2637.42 0.755 927.990 0.458  
 1 50000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack } 11448230 3232.6902 0.755 1165.920 0.825  
 1 50000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 11274062 2928.5193 0.735 1192.950 0.854  
 1 50000 dist=Dot k=30 ef=100 start_candidates=1 EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 9320067 1711.8726 0.782 832.640 0.542

# ensemble with 100k and 3 trees still strong:

    rep    n         search_params                              params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean

1 100000 dist=Dot k=30 ef=100 start_candidates=1 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 23785441 10417.359 0.739 696.280 0.441  
 1 100000 dist=Dot k=30 ef=100 start_candidates=1 RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot } 20409823 5250.9585 0.676 995.940 0.481  
 1 100000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack } 25036311 7242.8833 0.708 1345.290 0.952  
 1 100000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 24880813 6813.5396 0.683 1277.920 0.996  
 1 100000 dist=Dot k=30 ef=100 start_candidates=1 EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 18940086 3375.9253 0.779 896.450 0.585

repeated 3 times, still the winner:
rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 3 100000 dist=Dot k=30 ef=100 start_candidates=1 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 23893575 10372.44 0.730 744.530 0.460  
 3 100000 dist=Dot k=30 ef=100 start_candidates=1 RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot } 20421819 5129.695 0.438 686.967 0.334  
 3 100000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack } 25278947 7239.7285 0.701 1311.687 0.897  
 3 100000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 25110173 6682.601 0.692 1345.310 0.922  
 3 100000 dist=Dot k=30 ef=100 start_candidates=1 EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 18940086 3357.9453 0.744 920.110 0.603

# weird result, non threaded essemble being much faster to build than threaded with same params:

    rep    n         search_params                              params                                                                                                                                                                   build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean

3 100000 dist=Dot k=30 ef=100 start_candidates=1 SliceParralelRayonHnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 24020831 7322.283 0.748 721.373 1.304  
 3 100000 dist=Dot k=30 ef=100 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 17786535 4551.148 0.746 901.790 1.327  
 3 100000 dist=Dot k=30 ef=100 start_candidates=1 EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 18940086 3800.4504 0.739 904.280 0.581

# very weird result comparing threading in vp tree ensemble and hnsw:

    rep    n       search_params         params                                                                                                                                                                   build_ndc    build_ms     recall_mean    ndc_mean    time_ms_

3 100000 dist=Dot k=30 ef=100 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 24081239 10723.906 0.733 732.603 0.451  
 3 100000 dist=Dot k=30 ef=100 SliceParralelRayonHnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 24166631 7952.477 0.729 749.727 1.255  
 3 100000 dist=Dot k=30 ef=100 Threaded EnsembleParams { n_vp_trees: 3, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 32140932 7127.4395 0.786 873.670 1.218  
 3 100000 dist=Dot k=30 ef=100 EnsembleParams { n_vp_trees: 3, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 33588342 5916.7437 0.784 871.843 0.572

Why do the threaded ones have a search time that is so much higher? Locks?

rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 3 100000 dist=Dot k=30 ef=100 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 24111228 10897.315 0.743 738.433 0.450  
 3 100000 dist=Dot k=30 ef=100 SliceParralelRayonHnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot } 24004253 7827.6714 0.727 748.290 1.309  
 3 100000 dist=Dot k=30 ef=100 Threaded EnsembleParams { n_vp_trees: 3, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 32140932 6921.405 0.796 874.393 1.238  
 3 100000 dist=Dot k=30 ef=100 EnsembleParams { n_vp_trees: 3, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN } 33588342 6002.6323 0.781 880.523 0.584

# a small indication that multi-ef or kxk search could be a tiny bit valuable for (ef or k low e.g. 3). But multi ef not worth the runtime cost.

    rep    n        search_params                              params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean

1 80000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 512, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack } 27001734 4380.886 0.724 1168.000 0.596  
 1 80000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 512, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 1, only_n_chunks: None, distance: Dot, stitch_mode: MultiEf } 27265480 4464.5957 0.751 1282.170 0.658  
 1 80000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 512, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 3, only_n_chunks: None, distance: Dot, stitch_mode: MultiEf } 37687557 7152.831 0.784 1079.360 0.585  
 1 80000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 512, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 10, only_n_chunks: None, distance: Dot, stitch_mode: MultiEf } 62111514 25400.262 0.693 993.150 0.775  
 1 80000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 512, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 3, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 25977874 6188.2544 0.754 1285.780 1.261  
 1 80000 dist=Dot k=30 ef=100 start_candidates=1 StitchingParams { max_chunk_size: 512, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 10, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch } 25680941 5887.3853 0.730 1217.950 0.976

# ensemble params and hnsw roughly same recall 0.94:

rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 1 80000 dist=Dot k=30 ef=100 start_candidates=1 EnsembleParams { n_vp_trees: 6, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 20, m_max_0: 40, level_norm: 0.3, distance: Dot, strategy: BruteForceKNN } 46022016 8506.6045 0.944 1820.800 1.153  
 1 80000 dist=Dot k=30 ef=100 start_candidates=1 SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 20, m_max_0: 40, distance: Dot } 35582876 15879.692 0.942 1271.520 0.740

# ensemble method does not seem to profit from vp tree n candidates heuristic:

    rep    n         search_params    params                                                                                                                                                                            build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean

1 100000 k=30 ef=100 EnsembleParams { n_vp_trees: 6, max_chunk_size: 512, same_chunk_m_max: 16, m_max: 20, m_max_0: 40, level_norm: 0.3, distance: Dot, strategy: BruteForceKNN, n_candidates: 0 } 130930650 25085.672 0.954 1657.500 1.144  
 1 100000 k=30 ef=100 EnsembleParams { n_vp_trees: 6, max_chunk_size: 512, same_chunk_m_max: 16, m_max: 20, m_max_0: 40, level_norm: 0.3, distance: Dot, strategy: BruteForceKNN, n_candidates: 20 } 135747150 22658.156 0.947 1696.580 1.186  
 rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 1 100000 k=30 ef=100 EnsembleParams { n_vp_trees: 4, max_chunk_size: 512, same_chunk_m_max: 16, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN, n_candidates: 0 } 83846856 14772.367 0.956 1823.150 1.357  
 1 100000 k=30 ef=100 EnsembleParams { n_vp_trees: 4, max_chunk_size: 512, same_chunk_m_max: 16, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN, n_candidates: 3 } 84115776 15751.248 0.948 1828.930 1.261

# for stitching, selecting multiple vp trees also does not provide any benefit

    rep    n         search_params    params                                                                                                                                                                                                                   build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean

1 100000 k=30 ef=100 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack, n_candidates: 0 } 24919580 7939.873 0.703 1325.510 0.980  
 1 100000 k=30 ef=100 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack, n_candidates: 20 } 27392371 7983.5586 0.698 1341.950 1.132  
 1 100000 k=30 ef=100 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack, n_candidates: 40 } 28541031 8654.287 0.703 1295.570 1.025

### maybe tiny benefit for more vptree candidates

Start experiment stitching_n_candidates_n=200000_queries_n=100
rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 1 200000 k=30 ef=100 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack, n_candidates: 0 } 56489457 17794.785 0.658 1403.620 0.990  
 1 200000 k=30 ef=100 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack, n_candidates: 20 } 60550408 17874.648 0.660 1350.100 0.933  
 1 200000 k=30 ef=100 StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack, n_candidates: 40 } 63165243 18093.848 0.679 1367.720 0.968

### could RNN instead of brute force help?? from max_chunksize 512 and more seems faster!

        Finished searches for model in 6.715119 secs

rep n search_params params build_ndc build_ms recall_mean ndc_mean time_ms_mean  
 1 100000 dist=Dot k=30 ef=60 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 6, max_chunk_size: 256, same_chunk_m_max: 10, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN, n_candidates: 0 } 64281864 2907.5427 0.917 1402.061 0.557  
 1 100000 dist=Dot k=30 ef=60 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 6, max_chunk_size: 512, same_chunk_m_max: 10, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN, n_candidates: 0 } 122281602 5689.146 0.932 1349.180 0.523  
 1 100000 dist=Dot k=30 ef=60 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 6, max_chunk_size: 1024, same_chunk_m_max: 10, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN, n_candidates: 0 } 238872060 16707.848 0.944 1298.686 0.486  
 1 100000 dist=Dot k=30 ef=60 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 6, max_chunk_size: 2048, same_chunk_m_max: 10, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN, n_candidates: 0 } 472648566 70374.47 0.946 1231.675 0.567  
 1 100000 dist=Dot k=30 ef=60 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 6, max_chunk_size: 256, same_chunk_m_max: 10, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: RNNDescent { o_loops: 2, i_loops: 3 }, n_candidates: 0 } 64561824 7428.4067 0.911 1481.722 0.630  
 1 100000 dist=Dot k=30 ef=60 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 6, max_chunk_size: 512, same_chunk_m_max: 10, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: RNNDescent { o_loops: 2, i_loops: 3 }, n_candidates: 0 } 71836660 4802.4937 0.930 1461.148 0.558  
 1 100000 dist=Dot k=30 ef=60 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 6, max_chunk_size: 1024, same_chunk_m_max: 10, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: RNNDescent { o_loops: 2, i_loops: 3 }, n_candidates: 0 } 77393482 5455.635 0.940 1443.267 0.561  
 1 100000 dist=Dot k=30 ef=60 start_candidates=1 Threaded EnsembleParams { n_vp_trees: 6, max_chunk_size: 2048, same_chunk_m_max: 10, m_max: 20, m_max_0: 40, level_norm: 0.0, distance: Dot, strategy: RNNDescent { o_loops: 2, i_loops: 3 }, n_candidates: 0 } 81987581 12881.258 0.947 1459.626 0.670

### really weird multithreading speedup results on laptop:

n,dims,k,ef,start_candidates,params,build_secs,build_ndc,search_ms,search_ndc,search_recall
100000,768,30,60,1,"Hnsw_jpboth {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: False}",18.488778114318848,,0.5959329605102539,,0.8963433333332845
100000,768,30,60,1,"Hnsw_vecnn {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: False}",12.094955444335938,45527724.0,0.40775606632232664,932.8848,0.9129833333332908
100000,768,30,60,1,"Hnsw_jpboth {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: True}",3.421511650085449,,0.5859877347946167,,0.8937999999999523
100000,768,30,60,1,"Hnsw_vecnn {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: True}",8.346753597259521,45422597.0,0.40857391357421874,934.4684,0.9116733333332946
100000,768,30,60,1,"Hnsw_vecnn {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: False}",11.735878705978394,45527724.0,0.3286648511886596,932.8848,0.9129833333332908
100000,768,30,60,1,"Hnsw_jpboth {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: False}",18.705764055252075,,0.4428390026092529,,0.8976099999999564
100000,768,30,60,1,"Hnsw_vecnn {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: True}",2.51346492767334,45243734.0,0.3265370845794678,931.8262,0.9133033333332907
100000,768,30,60,1,"Hnsw_jpboth {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: True}",4.504664182662964,,0.4431221723556518,,0.8978666666666193
100000,768,30,60,1,"Hnsw_jpboth {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: False}",18.986008405685425,,0.5681008100509644,,0.8977866666666204
100000,768,30,60,1,"Hnsw_vecnn {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: False}",12.292339563369751,45527724.0,0.3969470024108887,932.8848,0.9129833333332908
100000,768,30,60,1,"Hnsw_jpboth {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: True}",10.863762617111206,,0.5704250574111938,,0.8944066666666185
100000,768,30,60,1,"Hnsw_vecnn {ef_constr: 20, m_max: 20, m_max0: 40, level_norm: 0.3, threaded: True}",10.346777200698853,45150173.0,0.40331599712371824,932.9695,0.9121999999999544
