#include <math.h>

double computeDistance(const double * feat1,const double * feat2,const int f1,const int f2, const int size)
{
      double descd = 0;
      double diff = 0;
      double previousdiff = 0;
      for(int v=9;v<size;v++){
        diff = (feat1[f1+v] + previousdiff) - feat2[f2+v];
        descd+=fabs(diff);
        previousdiff = diff;
      }

      return descd;
}