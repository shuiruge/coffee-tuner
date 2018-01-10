<TeXmacs|1.99.1>

<style|generic>

<\body>
  <section|Why>

  <section|How>

  <subsection|Notation>

  \;

  <subsection|Bayesian Approach>

  Let <math|n\<in\>\<bbb-N\><rsup|+>> the number of relavent features of
  making a good cup of coffee, e.g. the temperature of water;
  <math|x\<in\>\<bbb-R\><rsup|n>> the values of the features. Let <math|Y>
  the taste of coffee under a given <math|x>, which is either <math|0>
  (tastes bad) or <math|1> (tastes good), thus naturally is a random variable
  obeys a Bernoulli distribution with probability (confidence)
  <math|\<psi\>>, i.e. <math|Y\<sim\>Ber<around*|(|\<psi\>|)>>. Let <math|f>
  the model relates <math|x> and <math|\<psi\>>, depending also on paramters
  <math|w\<in\>\<bbb-R\><rsup|m>> for some <math|m\<in\>\<bbb-N\><rsup|+>>,
  i.e. <math|\<psi\>=f<around*|(|x;w|)>>.

  The Bayesian approach is as follow.

  <\theorem>
    We have

    <\equation*>
      p<around*|(|Y=1\|X=x|)>=\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<around*|(|W|)>><around*|[|
      f<around*|(|x,w<rsub|<around*|(|s|)>>|)>|]>,
    </equation*>

    where <math|w<rsub|<around*|(|s|)>>\<sim\>p<around*|(|W|)>> means that
    <math|<around*|{|w<rsub|<around*|(|s|)>>:s=1,2,\<ldots\>|}>> are sampled
    from <math|P<around*|(|W|)>>.
  </theorem>

  <\proof>
    By Bayesian formula,

    <\equation*>
      p<around*|(|Y=1\|X=x|)>=<frac|p<around*|(|X=x,Y=1|)>|p<around*|(|X=x|)>>.
    </equation*>

    Then by total probability formula,

    <\equation*>
      p<around*|(|X=x,Y=1|)>=<big|int><rsub|\<bbb-R\><rsup|m>>\<mathd\>w
      p<around*|(|X=x,Y=1,W=w|)>,
    </equation*>

    then Bayesian formula gives

    <\equation*>
      p<around*|(|X=x,Y=1|)>=<big|int><rsub|\<bbb-R\><rsup|m>>\<mathd\>w
      p<around*|(|Y=1\|X=x,W=w|)> p<around*|(|X=x,W=w|)>.
    </equation*>

    Since <math|x> and <math|w> are independent,
    <math|p<around*|(|X=x,W=w|)>=p<around*|(|X=x|)> p<around*|(|W=w|)>>. Put
    all together,

    <\eqnarray*>
      <tformat|<table|<row|<cell|p<around*|(|Y=1\|X=x|)>>|<cell|=>|<cell|<frac|p<around*|(|X=x,Y=1|)>|p<around*|(|x|)>>>>|<row|<cell|>|<cell|=>|<cell|<frac|<big|int><rsub|\<bbb-R\><rsup|m>>\<mathd\>w
      p<around*|(|Y=1\|X=x,W=w|)> p<around*|(|X=x,W=w|)>|p<around*|(|X=x|)>>>>|<row|<cell|>|<cell|=>|<cell|<frac|<big|int><rsub|\<bbb-R\><rsup|m>>\<mathd\>w
      p<around*|(|Y=1\|X=x,W=w|)> p<around*|(|X=x|)>
      p<around*|(|W=w|)>|p<around*|(|X=x|)>>>>|<row|<cell|>|<cell|=>|<cell|<big|int><rsub|\<bbb-R\><rsup|m>>\<mathd\>w
      p<around*|(|Y=1\|X=x,W=w|)> p<around*|(|W=w|)>,>>>>
    </eqnarray*>

    or simply,

    <\equation*>
      p<around*|(|Y=1\|X=x|)>=\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>><around*|[|
      p<around*|(|Y=1\|X=x,W=w<rsub|<around*|(|s|)>>|)>|]>.
    </equation*>

    And then insert <math|f> as

    <\equation*>
      p<around*|(|Y=1\|X=x,W=w|)>=f<around*|(|x,w|)>,
    </equation*>

    since <math|Y\<sim\>Ber<around*|(|f<around*|(|x,w|)>|)>>. So, in one
    word,

    <\equation*>
      p<around*|(|Y=1\|X=x|)>=\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<around*|(|W|)>><around*|[|
      f<around*|(|x,w<rsub|<around*|(|s|)>>|)>|]>.
    </equation*>
  </proof>

  What we want to find is a <math|x<rsub|\<ast\>>>, s.t.
  <math|p<around*|(|Y=1\|X=x|)>> (<math|=\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<around*|(|W|)>><around*|[|
  f<around*|(|x,w<rsub|<around*|(|s|)>>|)>|]>>) is maximized. Or say, we are
  searching

  <\equation*>
    x<rsub|\<ast\>>=<below|argmax|x><around*|{|\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<around*|(|W|)>><around*|[|
    f<around*|(|x,w<rsub|<around*|(|s|)>>|)>|]>|}>.
  </equation*>

  However, the only thing we have not known yet is the distribution of
  <math|W>. We are so humble that know nothing on how to make a good cup of
  coffee, so we use a flatten prior of <math|W>, i.e. <math|W\<sim\>Uniform>,
  with some support wide enough. We can obtain the posterior of <math|W> by
  inserting the data, i.e. a list of pairs <math|<around*|(|x,y|)>>, as the
  value of <math|Y> (the taste of a cup of coffee) given by some <math|x>. By
  feeding the data, we iterative gain the prior of <math|W>, which is the
  posterior in the previous iteration, as

  <\equation*>
    p<rsub|i+1><around*|(|W=w|)>=<frac|p<around*|(|Y=y<rsub|i>,X=x<rsub|i>\|W=w|)>
    p<rsub|i><around*|(|W=w|)>|p<around*|(|Y=y<rsub|i>,X=x<rsub|i>|)>>.
  </equation*>

  <\theorem>
    If define <math|g<around*|(|a,b|)>> as <math|b> if <math|a=1> and as
    <math|1-b> if <math|a=0>, and if initially use flatten prior, i.e.
    <math|p<rsub|1>=Const>, then we have, for data
    <math|D\<assign\><around*|{|x<rsub|i>,y<rsub|i>:i=1,2,\<ldots\>,N|}>>,

    <\equation*>
      p<rsub|N><around*|(|W=w|)>=c<around*|(|D|)>\<times\><big|prod><rsub|i=1><rsup|N>g<around*|(|y<rsub|i>,f<around*|(|x<rsub|i>,w|)>|)>,
    </equation*>

    or say,

    <\equation*>
      ln<around*|[|p<rsub|N><around*|(|W=w|)>|]>=<big|sum><rsub|i=1><rsup|N>
      ln<around*|[|g<around*|(|y<rsub|i>,f<around*|(|x<rsub|i>,w|)>|)>|]>+ln<around*|[|c<around*|(|D|)>|]>,
    </equation*>

    where <math|c<around*|(|D|)>> can also be seen as the normalization
    factor of <math|p<rsub|N><around*|(|W=w|)>> since Bayesian formula always
    ensures normalization of probability.
  </theorem>

  <\proof>
    By Bayesian formula and the independence between <math|X> and <math|W>,

    <\equation*>
      p<around*|(|Y=y<rsub|i>,X=x<rsub|i>\|W=w|)>=p<around*|(|Y=y<rsub|i>\|X=x<rsub|i>,W=w|)>
      p<around*|(|X=x<rsub|i>|)>
    </equation*>

    and

    <\equation*>
      p<around*|(|Y=y<rsub|i>,X=x<rsub|i>|)>=p<around*|(|Y=y<rsub|i>\|X=x<rsub|i>|)>
      p<around*|(|X=x<rsub|i>|)>,
    </equation*>

    thus

    <\equation*>
      p<rsub|i+1><around*|(|W=w|)>=p<rsub|i><around*|(|W=w|)>
      <frac|p<around*|(|Y=y<rsub|i>\|X=x<rsub|i>,W=w|)>|p<around*|(|Y=y<rsub|i>\|X=x<rsub|i>|)>>;
    </equation*>

    and since we have known in previous that
    <math|p<around*|(|Y=1\|X=x<rsub|i>|)>=\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<rsub|i><around*|(|W|)>><around*|[|
    f<around*|(|x<rsub|i>,w<rsub|<around*|(|s|)>>|)>|]>> and likewise
    <math|p<around*|(|Y=0\|X=x<rsub|i>|)>=\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<rsub|i><around*|(|W|)>><around*|[|1-
    f<around*|(|x<rsub|i>,w<rsub|<around*|(|s|)>>|)>|]>>, we finally get, if
    <math|y<rsub|i>=1>

    <\equation*>
      p<rsub|i+1><around*|(|W=w|)>=p<rsub|i><around*|(|W=w|)>
      <frac|f<around*|(|x<rsub|i>,w|)>|\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<rsub|i><around*|(|W|)>><around*|[|
      f<around*|(|x<rsub|i>,w<rsub|<around*|(|s|)>>|)>|]>>,
    </equation*>

    else (<math|y<rsub|i>=0>)

    <\equation*>
      p<rsub|i+1><around*|(|W=w|)>=p<rsub|i><around*|(|W=w|)>
      <frac|1-f<around*|(|x<rsub|i>,w|)>|\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<rsub|i><around*|(|W|)>><around*|[|
      1-f<around*|(|x<rsub|i>,w<rsub|<around*|(|s|)>>|)>|]>>.
    </equation*>

    \;

    After the first iteration, by <math|x<rsub|1>> and <math|y<rsub|1>=1>,

    <\equation*>
      p<rsub|2><around*|(|W=w|)>=Const <frac|f<around*|(|x<rsub|1>,w|)>|\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>Uniform><around*|[|
      f<around*|(|x<rsub|1>,w<rsub|<around*|(|s|)>>|)>|]>>=c<around*|(|x<rsub|1>,y<rsub|1>|)>
      f<around*|(|x<rsub|1>,w|)>.
    </equation*>

    Then the next iteration, suppose <math|y<rsub|2>=1> still,

    <\equation*>
      p<rsub|3><around*|(|W=w|)>=<around*|{|c<around*|(|x<rsub|1>,y<rsub|1>|)>
      f<around*|(|x<rsub|1>,w|)>|}> <around*|{|<frac|f<around*|(|x<rsub|2>,w|)>|\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<rsub|2><around*|(|W|)>><around*|[|
      f<around*|(|x<rsub|2>,w<rsub|<around*|(|s|)>>|)>|]>>|}>,
    </equation*>

    and re-define <math|c<around*|(|<around*|{|<around*|(|x<rsub|1>,y<rsub|1>|)>,<around*|(|x<rsub|2>,y<rsub|2>|)>|}>|)>\<assign\>c<around*|(|x<rsub|1>,y<rsub|1>|)>/\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<rsub|2><around*|(|W|)>><around*|[|
    f<around*|(|x<rsub|2>,w<rsub|<around*|(|s|)>>|)>|]>>, thus

    <\equation*>
      p<rsub|3><around*|(|W=w|)>=c<around*|(|<around*|{|<around*|(|x<rsub|1>,y<rsub|1>|)>,<around*|(|x<rsub|2>,y<rsub|2>|)>|}>|)>
      f<around*|(|x<rsub|1>,w|)> f<around*|(|x<rsub|2>,w|)>.
    </equation*>

    And if <math|y<rsub|2>=0>,

    <\equation*>
      p<rsub|3><around*|(|W=w|)>=c<around*|(|<around*|{|<around*|(|x<rsub|1>,y<rsub|1>|)>,<around*|(|x<rsub|2>,y<rsub|2>|)>|}>|)>
      f<around*|(|x<rsub|1>,w|)><around*|[| 1-f<around*|(|x<rsub|2>,w|)>|]>.
    </equation*>

    So, generally, if define <math|g<around*|(|a,b|)>> as <math|b> if
    <math|a=1> and as <math|1-b> if <math|a=0>, then for data
    <math|D\<assign\><around*|{|x<rsub|i>,y<rsub|i>:i=1,2,\<ldots\>,N|}>>,

    <\equation*>
      p<rsub|N><around*|(|W=w|)>=c<around*|(|D|)>\<times\><big|prod><rsub|i=1><rsup|N>g<around*|(|y<rsub|i>,f<around*|(|x<rsub|i>,w|)>|)>,
    </equation*>

    or say,

    <\equation*>
      ln<around*|[|p<rsub|N><around*|(|W=w|)>|]>=<big|sum><rsub|i=1><rsup|N>
      ln<around*|[|g<around*|(|y<rsub|i>,f<around*|(|x<rsub|i>,w|)>|)>|]>+c<around*|(|D|)>.
    </equation*>
  </proof>

  \;

  <\corollary>
    If data <math|D=<around*|{|x<rsub|BEST>,y<rsub|i>=1:i=1,2,\<ldots\>,N|}>>,
    then

    <\equation*>
      lim<rsub|N\<rightarrow\>+\<infty\>>
      <below|argmax|x><around*|{|\<bbb-E\><rsub|w<rsub|<around*|(|s|)>>\<sim\>p<rsub|N><around*|(|W|)>><around*|[|f<around*|(|x,w<rsub|<around*|(|s|)>>|)>|]>|}>=x<rsub|BEST>.
    </equation*>
  </corollary>

  <\proof>
    XXX <math|p<rsub|N><around*|(|W=w|)>=c<around*|(|D|)>
    <around*|[|f<around*|(|x<rsub|BEST>,w|)>|]><rsup|N>>
  </proof>

  <\algorithm>
    <\enumerate-numeric>
      <item>Set <math|p<around*|(|W|)>> by prior;

      <item>then iteratively:
    </enumerate-numeric>

    <\indent>
      <\enumerate-roman>
        <item>sample <math|<around*|{|w<rsub|s>:s=1,2,\<ldots\>,N<rsub|s>|}>>
        from <math|p<around*|(|W|)>>;

        <item><math|x<rsub|\<ast\>>=<below|argmax|x><around*|{|\<bbb-E\><rsub|w<rsub|s>><around*|[|f<around*|(|x,w<rsub|s>|)>|]>|}>>;

        <item>Make a cup of coffee by feature values <math|x<rsub|\<ast\>>>;

        <item>Taste the cupe of coffee;

        <item>Return your opinion as <math|y<rsub|\<ast\>>>;

        <item><math|D\<leftarrow\>D\<cup\><around*|(|x<rsub|\<ast\>>,y<rsub|\<ast\>>|)>>;

        <item><math|ln<around*|[|p<around*|(|W=w|)>|]>\<leftarrow\>ln<around*|[|p<around*|(|W=w|)>|]>+
        ln<around*|[|g<around*|(|y<rsub|i>,f<around*|(|x<rsub|i>,w|)>|)>|]>>;

        <item>fit <math|p<around*|(|W|)>> by variational inference which then
        replaces <math|p<around*|(|W|)>>;
      </enumerate-roman>
    </indent>
  </algorithm>

  <subsection|An Instant Model>

  \;
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|auto-3|<tuple|2.1|?>>
    <associate|auto-4|<tuple|2.2|?>>
    <associate|auto-5|<tuple|2.3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Why>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>How>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Notation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|2.2<space|2spc>Bayesian Approach
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2.3<space|2spc>An Instant Model
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>
    </associate>
  </collection>
</auxiliary>