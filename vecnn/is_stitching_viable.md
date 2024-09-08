Start experiment try it out_n=100000_queries_n=100
  rep    n         search_params    params                                                                                                                                             build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  1      100000    k=30 ef=50       { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, stitch_mode: RandomNegToRandomPosAndBack }    34681070     10122.47     0.636          1370.130    0.943  
  1      100000    k=30 ef=50       { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, stitch_mode: RandomSubsetOfSubset }           460245146    166108.77    0.297          1012.660    0.666  
  1      100000    k=30 ef=50       { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, stitch_mode: BestXofRandomXTimesX }           11731922     2124.815     0.061          559.530     0.375  
  1      100000    k=30 ef=50       { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, stitch_mode: DontStarveXXSearch }             33401271     8641.363     0.583          1456.140    0.959  

 RandomSubsetOfSubset just takes too long


  Start experiment try it out_n=30000_queries_n=100
  rep    n        search_params    params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  2      30000    k=30 ef=50       StitchingParams { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack }    9370907      2484.8535    0.710          1138.490    0.771  
  2      30000    k=30 ef=50       StitchingParams { max_chunk_size: 256, same_chunk_m_max: 40, neg_fraction: 0.3, keep_fraction: 0.1, m_max: 40, x: 4, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch }             9001535      2163.0542    0.642          1079.675    0.724  


  # params, with recall ca. 0.8 for 4 different models:


  Start experiment try it out_n=30000_queries_n=100
  rep    n        search_params                              params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  1      30000    dist=Dot k=30 ef=100 start_candidates=1    SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                                                                 6488720      1619.7968    0.798          684.430     0.304  
  1      30000    dist=Dot k=30 ef=100 start_candidates=1    RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot }                                                                       6043928      1411.5428    0.794          851.000     0.392  
  1      30000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 256, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 4, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack }    8039595      2014.0679    0.791          1062.670    0.836  
  1      30000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 256, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 4, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch }             7925060      1766.496     0.792          1103.270    0.818  

 # stitching with 160k

  Start experiment try it out_n=160000_queries_n=100
  rep    n         search_params                              params                                                                                                                                                                                                  build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  1      160000    dist=Dot k=30 ef=100 start_candidates=1    SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                                                                  40351556     19572.145    0.672          778.560     0.478  
  1      160000    dist=Dot k=30 ef=100 start_candidates=1    RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot }                                                                        32782910     8456.669     0.599          1126.340    0.577  
  1      160000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 256, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 10, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack }    47195121     13904.933    0.664          1330.320    0.923  
  1      160000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 256, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 10, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch }             47648609     11166.344    0.639          1387.370    1.002  


  # all about 0.74 recall:


   rep    n        search_params                              params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                                                                 11324875     3720.5044    0.745          698.180     0.410  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot }                                                                       10206200     2647.59      0.740          896.680     0.443  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack }    11348170     3127.3928    0.752          1185.370    0.857  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch }             11385147     2976.142     0.740          1251.780    0.891  


# ensemble is superior

    rep    n        search_params                              params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                                                                 11506835     4235.212     0.761          705.770     0.419  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot }                                                                       10153597     2637.42      0.755          927.990     0.458  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack }    11448230     3232.6902    0.755          1165.920    0.825  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch }             11274062     2928.5193    0.735          1192.950    0.854  
  1      50000    dist=Dot k=30 ef=100 start_candidates=1    EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }                                           9320067      1711.8726    0.782          832.640     0.542  


  # ensemble with 100k and 3 trees still strong:

    rep    n         search_params                              params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  1      100000    dist=Dot k=30 ef=100 start_candidates=1    SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                                                                 23785441     10417.359    0.739          696.280     0.441  
  1      100000    dist=Dot k=30 ef=100 start_candidates=1    RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot }                                                                       20409823     5250.9585    0.676          995.940     0.481  
  1      100000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack }    25036311     7242.8833    0.708          1345.290    0.952  
  1      100000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch }             24880813     6813.5396    0.683          1277.920    0.996  
  1      100000    dist=Dot k=30 ef=100 start_candidates=1    EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }                                           18940086     3375.9253    0.779          896.450     0.585  

repeated 3 times, still the winner:
    rep    n         search_params                              params                                                                                                                                                                                                 build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  3      100000    dist=Dot k=30 ef=100 start_candidates=1    SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                                                                 23893575     10372.44     0.730          744.530     0.460  
  3      100000    dist=Dot k=30 ef=100 start_candidates=1    RNNGraphParams { outer_loops: 2, inner_loops: 3, max_neighbors_after_reverse_pruning: 10, initial_neighbors: 12, distance: Dot }                                                                       20421819     5129.695     0.438          686.967     0.334  
  3      100000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: RandomNegToRandomPosAndBack }    25278947     7239.7285    0.701          1311.687    0.897  
  3      100000    dist=Dot k=30 ef=100 start_candidates=1    StitchingParams { max_chunk_size: 128, same_chunk_m_max: 20, neg_fraction: 0.4, keep_fraction: 0.0, m_max: 20, x: 2, only_n_chunks: None, distance: Dot, stitch_mode: DontStarveXXSearch }             25110173     6682.601     0.692          1345.310    0.922  
  3      100000    dist=Dot k=30 ef=100 start_candidates=1    EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }                                           18940086     3357.9453    0.744          920.110     0.603  


  # weird result, non threaded essemble being much faster to build than threaded with same params:

    rep    n         search_params                              params                                                                                                                                                                   build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  3      100000    dist=Dot k=30 ef=100 start_candidates=1    SliceParralelRayonHnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                        24020831     7322.283     0.748          721.373     1.304  
  3      100000    dist=Dot k=30 ef=100 start_candidates=1    Threaded EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }    17786535     4551.148     0.746          901.790     1.327  
  3      100000    dist=Dot k=30 ef=100 start_candidates=1    EnsembleParams { n_vp_trees: 3, max_chunk_size: 128, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }             18940086     3800.4504    0.739          904.280     0.581  


  # very weird result comparing threading in vp tree ensemble and hnsw:


    rep    n       search_params         params                                                                                                                                                                   build_ndc    build_ms     recall_mean    ndc_mean    time_ms_
  3      100000    dist=Dot k=30 ef=100  SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                                   24081239     10723.906    0.733          732.603     0.451  
  3      100000    dist=Dot k=30 ef=100  SliceParralelRayonHnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                        24166631     7952.477     0.729          749.727     1.255  
  3      100000    dist=Dot k=30 ef=100  Threaded EnsembleParams { n_vp_trees: 3, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }    32140932     7127.4395    0.786          873.670     1.218  
  3      100000    dist=Dot k=30 ef=100  EnsembleParams { n_vp_trees: 3, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }             33588342     5916.7437    0.784          871.843     0.572  

  Why do the threaded ones have a search time that is so much higher? Locks?

  rep    n         search_params         params                                                                                                                                                                   build_ndc    build_ms     recall_mean    ndc_mean    time_ms_mean  
  3      100000    dist=Dot k=30 ef=100  SliceS2HnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                                   24111228     10897.315    0.743          738.433     0.450  
  3      100000    dist=Dot k=30 ef=100  SliceParralelRayonHnswParams { level_norm_param: 0.3, ef_construction: 20, m_max: 8, m_max_0: 16, distance: Dot }                                                        24004253     7827.6714    0.727          748.290     1.309  
  3      100000    dist=Dot k=30 ef=100  Threaded EnsembleParams { n_vp_trees: 3, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }    32140932     6921.405     0.796          874.393     1.238  
  3      100000    dist=Dot k=30 ef=100  EnsembleParams { n_vp_trees: 3, max_chunk_size: 256, same_chunk_m_max: 16, m_max: 16, m_max_0: 16, level_norm: 0.0, distance: Dot, strategy: BruteForceKNN }             33588342     6002.6323    0.781          880.523     0.584  