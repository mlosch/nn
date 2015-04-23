#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSampledConvolution.c"
#else


static int nn_(SpatialSampledConvolution_updateOutput)(lua_State *L)
{
  return 1;
}


static int nn_(SpatialSampledConvolution_updateGradInput)(lua_State *L)
{
  return 1;
}

static int nn_(SpatialSampledConvolution_accGradParameters)(lua_State *L)
{
  return 0;
}

static const struct luaL_Reg nn_(SpatialSampledConvolution__) [] = {
  {"SpatialSampledConvolution_updateOutput", nn_(SpatialSampledConvolution_updateOutput)},
  {"SpatialSampledConvolution_updateGradInput", nn_(SpatialSampledConvolution_updateGradInput)},
  {"SpatialSampledConvolution_accGradParameters", nn_(SpatialSampledConvolution_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialSampledConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialSampledConvolution__), "nn");
  lua_pop(L,1);
}

#endif
