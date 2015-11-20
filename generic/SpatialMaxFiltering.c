#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxFiltering.c"
#else

static void nn_(SpatialMaxFiltering_updateOutput_frame)(real *input_p, real *output_p,
                                                      long nslices,
                                                      long iwidth, long iheight,
                                                      long owidth, long oheight,
                                                      int kW, int kH, int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    real *ip = input_p   + k*iwidth*iheight;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        long hstart = i * dH;
        long wstart = j * dW;
        long hend = fminf(hstart + kH, iheight);
        long wend = fminf(wstart + kW, iwidth);
        hstart = fmaxf(hstart, 0);
        wstart = fmaxf(wstart, 0);

        /* local pointers */
        real *op = output_p  + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        real maxval = -THInf;
        long tcntr = 0;
        long x,y;
        for(y = hstart; y < hend; y++)
        {
          for(x = wstart; x < wend; x++)
          {
            tcntr = y*iwidth + x;
            real val = *(ip + tcntr);
            if (val > maxval)
            {
              maxval = val;
            }
          }
        }

        /* set output to local max */
        *op = maxval;
      }
    }
  }
}

static int nn_(SpatialMaxFiltering_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int ceil_mode = luaT_getfieldcheckboolean(L,1,"ceil_mode");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;


  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4 , 2, "3D or 4D (batch mode) tensor expected");

  if (input->nDimension == 4) 
  {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }
  luaL_argcheck(L, input->size[dimw] >= kW && input->size[dimh] >= kH, 2, "input image smaller than kernel size");

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];

  oheight = iheight;
  owidth = iwidth;
  
  /* get contiguous input */
  input = THTensor_(newContiguous)(input);

  /* resize output */
  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

    nn_(SpatialMaxFiltering_updateOutput_frame)(input_data, output_data,
                                              nslices,
                                              iwidth, iheight,
                                              owidth, oheight,
                                              kW, kH, dW, dH);
  }
  else
  {
    long p;

    THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);

    input_data = THTensor_(data)(input);
    output_data = THTensor_(data)(output);

#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(SpatialMaxFiltering_updateOutput_frame)(input_data+p*nslices*iwidth*iheight, 
                                                output_data+p*nslices*owidth*oheight,
                                                nslices,
                                                iwidth, iheight,
                                                owidth, oheight,
                                                kW, kH, dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(input);
  return 1;
}

static void nn_(SpatialMaxFiltering_updateGradInput_frame)(real *gradInput_p, real *gradOutput_p,
                                                         long nslices,
                                                         long iwidth, long iheight,
                                                         long owidth, long oheight,
                                                         int dW, int dH)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    real *gradInput_p_k = gradInput_p + k*iwidth*iheight;
    real *gradOutput_p_k = gradOutput_p + k*owidth*oheight;

    /* calculate max points */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      for(j = 0; j < owidth; j++)
      {
        /* update gradient */
        gradInput_p_k[i*owidth + j] += gradOutput_p_k[i*owidth + j];
      }
    }
  }
}

static int nn_(SpatialMaxFiltering_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;

  /* get contiguous gradOutput */
  gradOutput = THTensor_(newContiguous)(gradOutput);

  /* resize */
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  if (input->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = gradOutput->size[dimh];
  owidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = THTensor_(data)(gradInput);
  gradOutput_data = THTensor_(data)(gradOutput);

  /* backprop */
  if (input->nDimension == 3)
  {
    nn_(SpatialMaxFiltering_updateGradInput_frame)(gradInput_data, gradOutput_data,
                                                 nslices,
                                                 iwidth, iheight,
                                                 owidth, oheight,
                                                 dW, dH);
  }
  else
  {
    long p;
#pragma omp parallel for private(p)
    for (p = 0; p < nbatch; p++)
    {
      nn_(SpatialMaxFiltering_updateGradInput_frame)(gradInput_data+p*nslices*iwidth*iheight, 
                                                   gradOutput_data+p*nslices*owidth*oheight,
                                                   nslices,
                                                   iwidth, iheight,
                                                   owidth, oheight,
                                                   dW, dH);
    }
  }

  /* cleanup */
  THTensor_(free)(gradOutput);

  return 1;
}

static const struct luaL_Reg nn_(SpatialMaxFiltering__) [] = {
  {"SpatialMaxFiltering_updateOutput", nn_(SpatialMaxFiltering_updateOutput)},
  {"SpatialMaxFiltering_updateGradInput", nn_(SpatialMaxFiltering_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialMaxFiltering_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialMaxFiltering__), "nn");
  lua_pop(L,1);
}

#endif
