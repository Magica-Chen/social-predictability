#include <map>
#include <cmath>

template <class seq_type>
int get_minlength(const seq_type& seq, int index)
{
  int n = seq.size() - index;
  int length = 1, first = 0;

  while (1)
  {
    if (first + length > index) return length;
    int i;
    for (i = 0; i < length; i ++)
    {
      if (seq[first+i] != seq[index+i]) break;
    }
    if (i == length)
    {
      while(1)
      {
        if (first+length > index || length >= n) return length;
        if (seq[first+length] != seq[index+length])
        {
          length ++;
          break;
        }
        length ++;
      }
    }
    first ++;
  }
}

template <class seq_type>
double S(const seq_type& seq)
{
  double sum = 0;
  int n = seq.size();
  for (int i = 0; i < n; i ++)
  {
    sum += get_minlength(seq, i);
  }
  return (n/sum)*log(n);
};

template <class seq_type>
double Sunc(const seq_type& seq)
{
  typedef std::map<typename seq_type::value_type, int> map_type;
  typedef typename map_type::iterator iterator_type;
  typedef typename map_type::value_type value_type;
  map_type counts;
  int count = 0;
  int n = seq.size();
  for (int i = 0; i != n; i ++)
  {
    count ++;
    iterator_type j = counts.find(seq[i]);
    if (j == counts.end()) counts.insert(value_type(seq[i],1));
    else j->second ++;
  }
  double S = 0;
  for (iterator_type i = counts.begin(); i != counts.end(); ++ i)
  {
    double p = double(i->second)/count;
    S -= p*log(p);
  }
  return S;
}


