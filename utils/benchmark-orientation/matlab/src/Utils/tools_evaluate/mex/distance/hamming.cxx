#include <math.h>


double computeDistance(const double* feat1,const double* feat2,const int f1, const int f2, const int size)
{
      double descd = 0;
      int diff = 0;
      for(int v=9;v<size;v++){//distance is computed here !!!!
        diff = (int)feat1[f1+v]-(int)feat2[f2+v];
        descd+=(diff == 0 ? 0 : 1);
      }

      return descd;
}
