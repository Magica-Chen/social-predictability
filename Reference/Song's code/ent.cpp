#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include "ent.h"

using namespace std;
typedef unsigned int uint;

typedef vector<string> t_seq;

int main(int argc, char* argv[])
{
  ifstream file(argv[1]);
  t_seq seq;
  while (true)
  {
    string data;
    file >> data;
    if (file.eof()) break;
    seq.push_back(data);
  }
  cout << "S = " << S(seq) << endl;
  cout << "Sunc = " << Sunc(seq) << endl;
  return 0;
}

