using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Tests;

public class PositionWiseFeedForwardTests
{
	[Test]
	public void ForwardTest1()
	{
		var inputs = new float[,]
		{
			{ -0.6075476972112264f, -0.12613641460642458f, -0.6846063587046353f, 0.9287147485934539f, -1.8444010340502783f, -0.46700242407710885f, 2.2924903431515506f, 0.48881005439557823f, 0.7102669906606637f, 1.05553444322415f, 0.05407310034578517f, 0.25795341634197144f, },
			{ 0.5882816497031765f, 0.8852442386126442f, -1.0170070242068927f, -0.13369303128332746f, -0.4381855013323405f, 0.4934434944281713f, -0.19900911984831327f, -1.274983607322147f, 0.2934941544378734f, 0.10895031182666172f, 0.03172678585903451f, 1.2726398630294582f, },
			{ 1.0714478994473615f, 0.41581801215857345f, 1.5506792313768636f, -0.31137891683028124f, -1.3792399073577473f, 1.3714087866765494f, 0.027711652280479462f, -0.3203995767435977f, -0.8461704050831389f, -0.43342891940622374f, -1.337034503851399f, 0.2091721706734753f, },
		};
		var fcWT = new float[,]
		{
			{ -1.424321301749055f, -0.17770715766502607f, 0.5428816787442755f, 0.5318740430245957f, 0.7017523620288093f, -1.692684648582905f, -0.06455678147092613f, -0.5531665203269047f, 0.22463386139820113f, 0.8578702220569501f, -0.8602597606806195f, -0.44994016470894355f, },
			{ -0.5534768527406204f, 0.6962776116386022f, -0.0844797040595787f, -1.0170234632092323f, 0.5983605127644394f, -0.619634653490464f, -0.1362632948860246f, -0.21255079196647503f, 0.3173195532137119f, 0.7684087326828745f, -1.0590564862509402f, -1.3397406149347593f, },
			{ 0.07479864396419998f, -1.188725096562167f, 1.2920150154310988f, -1.8480505423304856f, -0.5079949724231853f, 0.8743251683618662f, -0.17389384321325557f, -0.7334898338334107f, -0.21225177353307725f, 0.6035962840916762f, -0.7423757561145456f, 0.9484404405368387f, },
			{ -0.5056198334695845f, -0.3316968553727676f, -0.17671056902706545f, 0.17254907350770837f, 0.1361381164613738f, 1.7195190094755464f, 1.3216330610695413f, 0.3707289254133402f, -0.3135742042934178f, 0.2586647337329778f, -0.18639359743563763f, 1.5278355030832946f, },
			{ 1.0524077838685928f, 0.030076138124335197f, 1.6877871519898988f, 0.7862015331815178f, -1.4921145922656383f, 0.03429008899493663f, -1.568562411444584f, 1.030902596166464f, 1.6297357428103776f, -1.9500689755523521f, -0.3267171708721409f, -0.2569352606132207f, },
			{ 0.9714004076895887f, -1.1079151721093428f, -1.0466135423864138f, -0.07147002970413369f, 0.6116655365954832f, -0.8484343108566764f, -0.578639109874332f, 0.3039933362421069f, 0.17342488635935396f, -0.639845459047209f, 0.8043257067365511f, -1.3948854379864473f, },
			{ 0.07683153799891053f, -0.5499248981477401f, 0.6421202085543549f, 0.8363487706176722f, 1.3695275873478159f, 0.2688313634086204f, 0.4360061099481476f, 0.5730968143772961f, 0.11448989279160328f, -0.4133027378854977f, 0.2079762988661778f, 0.08272428384119018f, },
			{ -0.4350007751055339f, -2.0329095592585493f, -0.17296173715666704f, -0.19822888900551977f, -1.7027314169687207f, -1.7413772017521374f, -0.7529088041782768f, -1.218587764524949f, 0.08806335652883346f, 0.23156045672780104f, -0.03813704203076547f, -0.26019277713366634f, },
			{ 0.552994398084371f, 1.4079178035601656f, -1.112064974661889f, -0.09227256429260688f, -0.518780976726267f, 0.12649299011327006f, -0.9241022105161559f, -0.771470895369261f, 0.30395353760836197f, -0.7299980076875835f, 0.9202761532273347f, 0.4448073568932828f, },
			{ 0.26671630603188184f, 0.6331082634004426f, -0.020702781605266f, 0.8707258786625259f, -1.2347032313190298f, -0.15479409625000698f, -2.4370531318918807f, -0.3525092008989972f, -0.3794976246611468f, 1.6179641474143152f, -0.5438702845989606f, 0.8872979036124898f, },
			{ 0.008989408377556884f, 2.2127468853666246f, -1.8135217740605873f, -0.8391924798450838f, -2.8809945895530777f, 0.8069408900162098f, -1.588090506290977f, 3.437448945842229f, -0.8903746016846643f, -1.5170269887319645f, 1.868256965666641f, -1.3026176978305541f, },
			{ 0.6411027527491543f, -0.5266022787234301f, 0.20352189329759104f, 0.8232553742097967f, -0.015723283500143966f, -0.8913106344270815f, -0.9280105006278175f, 0.3577005395070786f, 0.8885192948141144f, 0.4560027636059438f, 1.7693499591706452f, -0.16682567480211946f, },
		};
		var fcB = new float[]
		{
			-1.7058216255457206f, -0.2203687476889161f, 0.5462762568466397f, 0.820008710189241f, -0.3036295429247585f, 1.7417510955520787f, 0.8760363335363743f, 0.7951173931042667f, -0.3915573588258649f, -1.1261498276195057f, 1.7610174849190638f, 1.000557907030824f,
		};
		var projWT = new float[,]
		{
			{ -2.1199454993193525f, 0.4814947567672656f, -0.5252576511252146f, -0.6389508598353981f, 1.2093030159753642f, -0.25405809162854026f, -0.5810976200189637f, -1.725481640996311f, 0.4114226098354317f, 0.00803837729851483f, 0.06840196955909968f, -0.8819300508997199f, },
			{ 1.9081588956350177f, 0.8352430787830553f, 2.452353799115599f, -0.24671840437059317f, 0.665395882779455f, -0.8110734114043393f, 0.44165881999423895f, -0.30352411340008106f, -0.8218397979251082f, -1.0042467871738512f, -0.081377033717167f, 0.46891312735180013f, },
			{ -1.0524780479559073f, 1.2260970414529409f, -1.7063270718776287f, -0.5112051308525494f, -0.615067284115001f, -1.632052987803717f, -1.5023985647972131f, -1.5312495983265948f, 0.1237474627276661f, 0.18260970663230996f, 0.7695594858133961f, -0.020052512019736943f, },
			{ 0.08132310056727626f, 1.721692829296525f, 0.4500601738114051f, -0.3301478915346678f, -0.12189264979065029f, -0.8755343534947387f, -1.6729413138073828f, 0.886166522305261f, -1.664130697223253f, 1.4446647480801036f, 0.2476576005583191f, 1.7117717624039657f, },
			{ 0.947082679707724f, -0.21137760948131554f, -1.0421403085239738f, -1.450177243003694f, 0.4139411771253432f, 0.35190973391424407f, 0.8960270137894901f, -0.0015840314358094257f, 1.5919560487987123f, 0.2143330678426898f, -1.107035550289962f, -0.6556114052572146f, },
			{ -0.5337167395615388f, -1.2159829549869632f, -1.4767444329587256f, -0.023982850194145554f, -1.1452624604601476f, 1.0756909562723351f, -0.4664442935037952f, -1.018887109100286f, 2.270219048944487f, 0.29789950888949596f, 0.41045308456957375f, 0.12779087760048746f, },
			{ 0.013596459246635705f, 0.6397264004476141f, -0.5187014145950604f, -2.9122368771414955f, -0.0418781122374516f, 0.4572594328274544f, 0.8366492414874737f, 0.6442599665402782f, -0.23358451766321306f, 1.8958582845807113f, 0.33561588327866165f, 1.5514042822732164f, },
			{ -1.4383804368907815f, -1.2475100940431836f, -0.017552426035479705f, -0.395186612989389f, 0.35195852969079555f, 0.8228248606148078f, -0.7610962589059204f, -0.2942813360748605f, -0.12686835651445302f, -0.40316962462457456f, -0.7331731221437654f, 1.1603451193338943f, },
			{ 1.013269776974541f, 0.031376958796576024f, 1.9510212228245518f, 0.5967495671010836f, -2.5100104397666585f, -0.485597629249781f, 0.34688458486773716f, 1.4481115104210351f, 0.8746482024008275f, 0.243855029428207f, -0.689567082491706f, 0.21228831986380087f, },
			{ -0.38062482187234425f, 0.5557689944751371f, 0.3680977157513193f, -1.8283525558427485f, -0.11960286447900124f, -0.9797176168682161f, -0.45746352271274304f, 0.15356169346655432f, 0.10349995783964763f, -0.7957882979015644f, -1.0924954408118952f, 1.0991403550505083f, },
			{ 0.30563644533717516f, 1.5563008461578025f, -0.9762992745125145f, -2.863944480520985f, 0.43344922752541937f, -0.15028512793835888f, 1.0077722439591061f, -0.014832362193619067f, 0.943187120207635f, -0.5411547141683226f, 0.21878402632250443f, 1.156262579961358f, },
			{ 1.189978221940916f, 1.2840826955465308f, 0.3543085317836314f, 0.6924712479739551f, -0.4296482748089386f, 1.955553946231751f, -1.4537872397248843f, 1.4909772366239311f, 0.19513738529771124f, -0.21251606852497812f, -0.9827791818131986f, 0.22512217760377715f, },
		};
		var projB = new float[]
		{
			0.5719586025044407f, 0.08814005309712146f, -0.15986960048433982f, -1.590915361019473f, -0.17005117085391674f, -0.3053427198752506f, -0.4945812152885449f, 0.20165514257771577f, 0.6139640053364958f, 0.9157572122194848f, 0.16014549438230447f, -0.827757385492154f,
		};
		var expected = new float[,]
		{
			{ -7.458950056557762f, -2.228060269481075f, -4.428455723568924f, 1.9748834047658004f, -9.527756549495649f, -2.2849551439881264f, -8.457310565561887f, -2.804808747325062f, 5.681080996271051f, -7.553267666184113f, -10.266015099646319f, 3.925783108831939f, },
			{ 0.09421724886753474f, -3.4824494154309114f, -1.4932168280559337f, -5.627767070785267f, 1.3376120475395037f, 8.90404053228461f, -4.620711984948749f, -2.8612628393023645f, 8.176172097015396f, -5.0663475872200845f, -4.472077046551609f, 0.5307090273513917f, },
			{ 2.006084522752147f, 15.988150303339298f, -12.001785801582908f, 2.006415508068647f, -11.316310243030065f, -13.241396287328454f, -8.752304013162789f, -1.3541975138529094f, -0.2790719940848029f, -6.066525215017684f, -11.497625481330436f, -2.0885033427492536f, },
		};

		var output = ArrayFactory.NewLayerOutput(inputs, projWT);
		PositionWiseFeedForward.Forward(inputs, fcWT, fcB, projWT, projB, output);
		Assert.That(output, Is.EqualTo(expected).Within(0.00001f));
	}
}
