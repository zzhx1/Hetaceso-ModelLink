# MindSpeed-LLM 单样本指令微调


MindSpeed-LLM支持指令微调，方案与<a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a>统一，在微调效果保持一致的前提下，MindSpeed-LLM可以表现出优异性能

【与<a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a>微调Loss对比】
<table>
<tr>
<th rowspan="1">Llama2-7b模型与<a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a>微调5个epoch后的loss对比图
<p align="center"> <img src="../../sources/images/tune_llama2_7b_MindSpeed-LLM_DeepSpeed_compare.png" height="270px" width="500px"> </p>
</th>
<th>
<th rowspan="1">Qwen-7b模型与<a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a>微调4个epoch后的loss对比图
<p align="center"> <img src="../../sources/images/tune_qwen_7b_MindSpeed-LLM_DeepSpeed_compare.png" height="270px" width="500px"> </p>
</tr>
</table>


【Llama2-7b微调后对话效果对比】
<table>
    <tr>
        <th rowspan="1">Question</th>
        <th rowspan="1">MindSpeed-LLM</th>
        <th colspan="1"><a href="https://github.com/hiyouga/LLaMA-Factory/tree/main">DeepSpeed</a></th>
    </tr>
    <tr>
        <th>What is the product of 6 and 2?</th>
        <th>The product of 6 and 2 is 12.</th>
        <th>The product of 6 and 2 is 12.</th>
    </tr>
    <tr>
        <th>Look up the boiling point of water.</th>
        <th>The boiling point of water is 100°C (212°F)..</th>
        <th>The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit).</th>
    </tr>
    <tr>
        <th>Generate a poem that expresses joy.</th>
        <th>I open my arms and accept life's challenge, 

Embracing all of life's uncertainty. 

A future of promise awaits,

One that I can find in my fate. 

A journey full of hope and discovery,

A journey that will never end in disappointment. 

For I have found the strength to keep going,

Researching and learning, ever knowing. 

The future is mine, and mine alone,

A future that I have to own,

And I will never be alone,

For I will have the courage to face all of my fears,

And the joy of life will be mine forever.</th>
        <th>I am so happy, so thankful and blessed,

To have a heart that's full of cheer,

My soul is on fire, my spirit runs free,

My love is warm, my courage is high,

My world is bright and my outlook is clear,

My heart is light, my spirit runs free.

I take each day with a positive outlook,

My energy is so high, I can't be beat,

My joy is real, my emotions run deep,

My soul is full and my heart does soep.

I am thankful for everything I have,

My life is precious and my values ​​true,

My hope is high and my spirit runs free,

My soul is full and my heart does soep.</th>
    </tr>
</table>
