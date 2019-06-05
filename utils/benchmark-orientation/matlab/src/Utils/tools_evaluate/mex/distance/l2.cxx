#include <math.h>

double computeDistance(const double* feat1,const double* feat2,const int f1, const int f2,const int size)
{
      double descd = 0;
      for(int v=9;v<size;v++){
        descd+=((feat1[f1+v]-feat2[f2+v])*(feat1[f1+v]-feat2[f2+v]));
      }
      descd=sqrt(descd);

      return descd;
}