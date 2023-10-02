using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using ToyGPT.NeuralNetwork.Layers;

namespace ToyGPT.NeuralNetwork.Tests;

public class TransformerBlockTests
{
	[Test]
	public void ForwardTest1()
	{
		var inputs = new float[,]
		{
			{ 1.4048395660218949f, 0.2211210412399543f, -0.14532730653059658f, 0.12319916869046948f, 0.6060269739057204f, 2.4227700144981212f, -1.9166085406265887f, -2.4225270901000844f, 0.6462942175522528f, 0.20150063672609f, -0.15671318238190332f, 0.7720457639044952f, },
			{ -1.35760992080276f, 0.7017173797660419f, 1.0533246600640918f, 0.2699069220028918f, -0.9708542721112963f, 1.047881787575985f, 1.2191810013041187f, -0.051679982931552834f, -1.1082717118362218f, 0.33130129990258456f, -1.3617634904761577f, 0.113779684174314f, },
			{ 0.5600109388079717f, 1.9580267991632618f, 2.0635263129583072f, 0.5809961982196509f, 1.0010068260817988f, 0.5377727285953665f, -0.9240831491190826f, -2.661530452969075f, 0.7867800199639358f, 0.4978533386788726f, 0.2595225502741862f, -0.8932681095536522f, },
		};
		var mhaUpWT = new float[,]
		{
			{ -0.9757826484021094f, 0.031460301746884065f, -1.4382864361320242f, 0.8318218757844626f, -0.09020238467477319f, 1.2578885021887445f, 0.14576379476556722f, 1.282904422903443f, 0.3726615959433017f, -0.8845626081619709f, 0.845926605053272f, 0.4081860866393206f, },
			{ -1.0372388625167075f, -0.24275683793268965f, 0.5762541465605887f, -0.8747273508848433f, 0.7952017534298856f, -1.1938592602770082f, 0.7618077450112181f, -1.039165102646771f, -1.2315813473908785f, 1.1399728367283268f, 0.09736514210421367f, 0.5595325719618383f, },
			{ 0.4067829406180238f, 2.207796957281153f, -0.8412062702651957f, 0.7949054856607302f, -0.31231351810638863f, 1.2685214183677391f, 0.10625085005651322f, 2.3650508707113844f, 1.2194961105747864f, 0.014200344281134935f, -0.8458607165811352f, -0.27187555086361836f, },
			{ -0.668680234368785f, -0.10536533187850584f, 0.295393280496803f, -1.5680011901427382f, 1.8554233810017513f, -2.191179777007931f, 0.6478163506954043f, -0.7529248125191093f, -0.4979731882926542f, -0.3953676877358709f, 0.2910219307463887f, 0.7242707417560803f, },
			{ 0.602839932940179f, 0.25959325872360983f, -0.8349864869298687f, 0.7827172259948174f, 0.915154938872749f, 0.9807609018166064f, 0.18283434611156257f, -0.7879102753229794f, 1.8055856652981674f, -0.9044016122974973f, -1.286757340309651f, 0.21605403683552266f, },
			{ -0.6147593873858278f, 1.0013227893439913f, -1.155421858529671f, -1.3567912969686264f, -0.838210797876499f, 0.9843561425179722f, 1.0554516506311233f, -0.6880563645120515f, 1.1420692635738863f, -0.6944435668223139f, 1.8076707521913589f, -0.3776410435900749f, },
			{ -0.8883760188822638f, -1.288804094091532f, 1.0337201257342803f, -1.5469326773145466f, 0.5134484598167697f, 0.12524466332823936f, -0.46990251361558044f, -0.15888826871406955f, 0.4999939723811579f, -0.5068884777197862f, 0.7904644225280115f, 0.09764836063336532f, },
			{ 1.2912460207864653f, -1.1780275385749872f, 0.551383772583689f, 0.6767065511469874f, -0.6020779143755571f, 0.010128714223852371f, 0.46097106273928384f, -1.0288247537307837f, -1.18898247198511f, 0.025060240131028883f, -0.5612934861015395f, -0.7385655671284742f, },
			{ -0.6977652898617516f, -1.2990543409597903f, -0.486205858338544f, 1.0486938825995022f, 0.08092707549440098f, 0.6612295896901904f, -1.4842835914200305f, 0.2350741095539362f, -0.03571495309344784f, -1.8326541387602522f, -0.9277131384217944f, 2.7222047636443905f, },
			{ -0.7385429975639791f, 0.8445998772961891f, 0.9361642142601122f, 0.6091098046736989f, -0.6844738286254944f, 0.04407653599852609f, 1.272279983930843f, 0.9107657952179768f, -0.2115201203054116f, -0.2913148222483837f, 0.6326502621732275f, -1.1654850485059487f, },
			{ 0.9503890266556926f, -1.2639432842645517f, 0.3614266200988144f, 0.10694020657233f, -0.21873127719308139f, -1.3047792453246747f, 0.5987208613526008f, 0.7346278241211279f, 0.15111206518232573f, -1.3344455890396536f, -0.8674220619825186f, -0.4854424907957445f, },
			{ 0.09887465535315301f, -0.1425867658516095f, 0.4111935748210536f, 0.5918389889580524f, -0.2885918931503087f, 0.2378486143029786f, -1.9851237295587498f, 0.410289353352635f, 0.6058531915164858f, -0.170432997390159f, -1.0366324112201115f, -0.1069532009887878f, },
			{ -0.3987488813853751f, 0.11576594798261962f, 1.08114064808586f, -0.7539723522015694f, 0.08531271736806367f, 0.6601271596518861f, -0.34906542223310066f, -0.7994425623437683f, -0.3560199113522886f, -1.154857143879731f, 1.0602958585269227f, -1.0055401325524689f, },
			{ 0.7408233064907825f, 0.7812561644633251f, -0.8389735593565505f, -0.9331483321244105f, -0.31760733679344416f, -0.5929493893754599f, 1.0178089110225295f, 0.1291055170210159f, 2.159445540139499f, -0.2801061154462335f, 0.1284792555420747f, 0.7502438600776864f, },
			{ -1.1605333525129826f, -0.42125569010008007f, -0.6470653872719412f, -1.6010628167940573f, 0.5133083423636391f, 0.7477502381037036f, -0.4607815737626208f, -0.7030322727887633f, 0.453156540154111f, -1.5753811648937286f, 1.3379918663106654f, -1.6215567558050408f, },
			{ -1.047042546714536f, -1.019924352059101f, -0.7639965011514752f, 0.08667601216797886f, -1.4739409353145778f, 1.1380104184246487f, -1.3016255379685036f, -1.9772720284033238f, 0.7310793119530219f, -0.6662837982389676f, -2.1466367383246734f, 0.8263662055149639f, },
			{ -1.7538845380500292f, -0.11844941155905295f, -0.5107584563064026f, 1.3136787371104872f, -1.3000596250351089f, -1.909372285971712f, -0.18502529869701737f, 1.0422541301001587f, -0.4649488854927612f, 0.7978205660556197f, -1.4684068622453137f, 0.3813495704580528f, },
			{ -0.9723995173305972f, -0.5303136927775709f, 1.6805289869784383f, 0.9064791273020728f, 1.7682493689764651f, 0.5326537911428469f, 0.9495977624676326f, 1.114529200324642f, -0.07432945467701053f, -0.2597207445767238f, -1.188550948854088f, -1.3943168001439017f, },
			{ -0.08349147868966367f, 1.3178772803465737f, -0.5295062522454959f, -1.009730888441563f, 0.05861223908567602f, -0.8565581007257592f, -0.5843199369441685f, -1.353610995315964f, -1.177977988907862f, 0.28058803595016457f, 1.9932432863885479f, 1.3935062678769952f, },
			{ 0.6107286944322622f, -0.48923240465913753f, -0.7650412991427104f, -1.5138500513600692f, 0.9103996003802383f, -0.08725998203398576f, -1.2214380165768968f, -0.42222111712212057f, 1.3960180438050214f, -0.009260129580211253f, -1.6926344611850468f, 0.03835141461322277f, },
			{ 1.1030821750720026f, -0.9813662598651746f, -0.07768411713113607f, 0.5399803906418517f, 2.3106869375169072f, -0.0594284247344643f, 1.9782362233475914f, 0.6564517190311023f, 1.598114777987366f, -0.9319089363028731f, 0.7430507053483316f, -0.493503602934977f, },
			{ -1.111614252555771f, -0.4916275855452904f, 0.5895842684520194f, -1.1624314442915853f, 0.5635221240142796f, -0.6603878430066494f, -1.8193565287147104f, -0.527480439928708f, 0.654512600470273f, 0.12127345446816412f, -0.0823656518215448f, -0.3327257716938209f, },
			{ 0.5194055215597513f, 0.8921490476092875f, -1.575572503526757f, -1.1566165852549488f, -0.34147997939714003f, -0.8135221728086949f, 0.8378515662800614f, 1.4664766973917f, -0.8241455267376132f, 0.32477595415682453f, 0.639865961142071f, 0.053283071019171103f, },
			{ 0.7919447063564294f, 0.45884112358914164f, -1.5836246153530276f, -0.03872044313860875f, 2.164525108855499f, -0.033420780625122395f, -0.4454365124468052f, 0.33656287673821517f, -1.4191951803368805f, 0.9420296794065668f, 0.7721960383631337f, 1.6672797814779832f, },
			{ -0.5902848232677586f, -1.15268552898212f, 0.33633472467944814f, 2.413285456638592f, -0.6935127815848694f, -0.8836921544033858f, 0.660910784832929f, 1.8156578884059769f, -0.25941926170852597f, -0.3517708852582649f, 0.3316525868595974f, 0.4434390780833072f, },
			{ 0.34047864858380433f, -0.39924852081282564f, -1.5006694593675185f, 0.42508227976385404f, 0.9603827095746866f, -1.2278022253222878f, -1.1138077042914356f, -0.3823730901556923f, 0.14355615595497617f, 0.7283027597431003f, -0.274583571200714f, -1.311898643321412f, },
			{ 2.696975648111834f, 0.2998652691733106f, -1.9941434800238316f, -1.8622923313560982f, -0.044576845394964884f, 0.04084698137673948f, 1.1537623384168891f, -0.4791768161735909f, 0.432976116551657f, 0.8615570323911486f, 0.5929388882215894f, -0.6804665112446168f, },
			{ -0.7613684144884635f, 0.04169341155129364f, -0.419961500851197f, 1.8205407980161834f, 0.9400872476999291f, 0.7680583012887242f, 1.8818103947452045f, -0.04048024718572368f, -1.5188787828881485f, -0.8205642932237706f, -0.9286712605331005f, 1.3558065254161968f, },
			{ 0.9918368692289408f, -2.5519638292343076f, 0.4663495650516257f, -0.6459108368545231f, -0.9454708214464279f, 0.6861285318824667f, 1.3030524011792954f, -1.761945443136933f, 0.2406038227984362f, 0.6220625779143502f, 0.21772208495858905f, -0.29038074006100534f, },
			{ -0.6397992931000751f, 1.2035013695362182f, 0.8880549800129001f, 0.7356794083228582f, -0.12110583181964191f, 1.5846897366116117f, 0.3537939352578454f, 1.609461414054646f, 0.8496647473657536f, 0.6560854567924949f, 0.04672128386519929f, 1.569532640323734f, },
			{ 0.3578373744751262f, 0.22374946242301277f, 2.714643156086997f, -1.1721565535603962f, 0.3648519152476801f, -0.19699728542752365f, -0.7447326742343287f, -1.4041658265560009f, 2.130703357989338f, 1.0464774374874417f, -0.2431054805588302f, -1.0645514172361439f, },
			{ 0.6702027284384305f, 0.39818641481713113f, -0.5222292138096258f, -0.08510714693579632f, 0.7080915586909136f, 0.9962836775420127f, 1.6453373281209012f, 0.40678403878084396f, -1.3904010837032068f, -0.1621843225788774f, -1.318680469736072f, -0.22612558740406544f, },
			{ -0.11508806503712327f, -0.3557544921212467f, -0.13518833076890371f, 1.3316688222477853f, -0.09073149049871229f, -0.8097591448575848f, -2.2440410287981893f, 0.275116333319733f, -1.9622713113884689f, -1.539695517551735f, -1.1150887293530194f, 0.6473040165466099f, },
			{ 0.5317722224214109f, -0.42042191768250614f, 1.4104978581037901f, -0.45783709007839113f, 0.22081608446690498f, -1.123203179547462f, 0.03085965551319594f, -0.20338395051885466f, -2.699240482138786f, 0.7117514569660489f, 0.3562209687728607f, 1.3050432897648339f, },
			{ 1.525711097587063f, -0.9030011312355943f, -0.46678009080428196f, 0.8527830121858472f, -0.20309997363322937f, -1.1641561956882809f, 0.4200923834648271f, 0.09337862784254172f, -1.8928208164693612f, -0.6057086717411958f, -0.46401932709679844f, -0.6408670130843448f, },
			{ 0.11677377743918403f, -0.17717761790325268f, 1.9336639073452708f, 0.047302687344241784f, -1.4817092230024138f, 0.023348726691668986f, -1.1597627653192881f, -2.4388766937417525f, -0.13825931948190096f, -0.09110921775861114f, -0.10130136294057249f, -1.928126355014126f, },
		};
		var mhaUpB = new float[]
		{
			0.0812066577046453f, 0.31123711907806934f, 0.4241494962612009f, -0.7467742673000078f, -0.052534056488043344f, -0.025530984666415256f, 0.11628303794813274f, -0.13683776876189133f, 1.6498079353089505f, -1.230893595554221f, -0.9865601978465897f, 1.2357531953272132f, -0.7953716624149616f, -0.957881238457113f, 0.767828888488452f, 1.19966572397227f, 0.6054348692641274f, 1.269901780147245f, -0.8259266914139595f, 0.5294915217251266f, -0.2524992102235132f, -1.1604422596898822f, -0.14550268724448756f, 0.6873638757302297f, 1.6442252167166826f, -1.318306487969778f, 0.0021532865342902188f, -0.32234776053883424f, -0.8064845106727236f, 0.4992468058718149f, -2.3017374350436515f, 1.9608033076378584f, -0.5752935965900533f, 0.29895870046678935f, -1.395633816382062f, -1.1594646440374363f,
		};
		var mhaDownWT = new float[,]
		{
			{ -0.6105754544598191f, 1.1329608449727624f, 0.9793140056209689f, 1.1297285220116526f, 2.516370244810218f, 0.06795614809605575f, 1.4414926971500432f, 0.6985209892869181f, 0.7648631879701286f, 1.2335612013548611f, 0.09342434054744594f, 1.1261611153170066f, },
			{ 0.19943649389152687f, -1.4248271740047629f, -1.3010213770856365f, -0.5196305205990148f, 0.1135724076016738f, 0.1467724823182659f, -0.47812863293434793f, -0.07129885072026086f, 0.6492652168370185f, 0.7105406322759716f, -0.31556039841786987f, 1.4802777120032455f, },
			{ 0.046392185575306326f, 1.219441754080207f, -0.495413134879426f, 0.7725557908346777f, 0.023194499835290316f, -0.324131196631414f, 0.2733878540877269f, -0.576498948842148f, 2.1808662489524746f, 1.0535382349887072f, -0.691639407852137f, -0.49791425358207836f, },
			{ 0.7301729487329687f, -0.41068785086626436f, 0.5344484347175091f, -0.35448996315577574f, -0.01449427235703496f, -1.595407303043764f, 1.340046775734947f, 1.1992514943167112f, 0.9307940568746775f, -0.4393691832346279f, 0.7819468069474637f, 1.0664099863238212f, },
			{ -1.8253730733918203f, 0.08030397667215446f, 0.8428791377524876f, -0.41644779478034727f, 1.1818457010596568f, 1.7622155463724132f, 0.6942055181818323f, 1.2717731736926745f, 0.14027756559182206f, -0.4078429854444219f, -0.17233621797734958f, -0.24274396795241435f, },
			{ -0.9806232330746136f, 0.180643606834809f, 1.074091237080534f, -0.9050336780014265f, -0.5095149823331141f, -1.7658379817359502f, 0.06866427677198705f, 0.9672018947443416f, 0.5319083548611743f, 1.670556685787498f, -0.7285862469241657f, 1.36461436857686f, },
			{ -0.3586084379296227f, -0.6526009072618306f, -1.871768901691402f, -0.31434641749364167f, 0.17700448157981288f, 1.006478255021648f, -0.20891159747443863f, 2.655969534742597f, 0.405888443210953f, 1.2361591794859053f, 0.5119084842293691f, 1.078161604532672f, },
			{ -1.1976422542821699f, -0.8114476812390344f, -0.42516177161640295f, -0.3046290293062366f, 0.21276140014866757f, -1.3984895957268875f, 0.16210734261112178f, -1.3778577358729323f, -0.49719341046038257f, -0.4251312211970695f, 0.3839942813811296f, -0.9805046882054836f, },
			{ 0.35650930749519405f, 0.6619800964672956f, -0.17891745889911836f, -0.5498572728242273f, 1.4040239298894168f, 0.2528665903308632f, -1.701062407928664f, 0.8975072945131127f, -0.5921817342527574f, 1.0328613323688383f, 1.3345484117901791f, -0.12025831872304982f, },
			{ -0.07654247757705293f, -0.30623936715101274f, -1.3250910465754646f, -1.4143949455213143f, -0.3364275546860459f, 1.3379558303224304f, 1.1785993941225468f, 0.15131786969843306f, 1.428968274322107f, -0.003988955951440034f, -1.236326193635096f, 0.07505065197923008f, },
			{ 0.7198913477327727f, 1.0888244311949282f, 2.5638373881795142f, -2.5125398777056867f, 1.208433652389282f, -0.7589759859341014f, 1.6686558307154264f, -0.4660199461115777f, -2.543872399627137f, -0.16335239136330224f, -0.3146180384777539f, 0.601844635017835f, },
			{ -0.5352908103055255f, -0.4973829118599899f, -0.9038952045181554f, -0.8139720489900582f, -2.138064761613677f, -0.6889602873037136f, -0.39154187896393167f, -1.2042232931974937f, -0.6214713712005219f, -0.20443130625563946f, 0.5333546829860496f, -1.0474568931717974f, },
		};
		var mhaDownB = new float[]
		{
			-1.0845132399387032f, 0.061797903440205186f, -0.6692414412068934f, 0.3745706952918023f, 0.3051444362016838f, -0.3748740319083904f, -0.717267993675048f, -1.32497531035672f, -1.1762079340547016f, -0.603144687002087f, -0.1853898341202884f, -0.2393868407509069f,
		};
		var mhaLnG = new float[]
		{
			-1.6820983274910275f, -0.13406921131974245f, 0.6380698185959822f, 0.40631525957640763f, 0.9619103932447566f, 0.07763164322378488f, -0.7712666800795523f, 0.2916898517302754f, -2.707377114615606f, 0.4925675853033617f, -0.5298091541023516f, 0.6225508371359193f,
		};
		var mhaLnB = new float[]
		{
			-0.9315564164244681f, -0.2595941227546275f, 0.15559329059396435f, -0.21968359385086703f, 0.04694989540014483f, -1.0599359491946965f, 0.8082603441460087f, 0.4328639232854981f, -0.07791207065613766f, -0.8527396647365578f, -0.5486179025859045f, -0.5974323340193917f,
		};
		var ffnUpW = new float[,]
		{
			{ -0.174704523476464f, 0.8706011012003229f, 1.2213758734436149f, 1.2591847810913372f, 0.6347348376770577f, -0.2709784304231168f, 1.6530684866726801f, -0.6616870348454171f, 0.0749094476183263f, 0.5776793590829061f, -3.054633398804808f, 0.3644110148695365f, },
			{ 2.2855349175381887f, -1.0674892997104828f, -0.2629681927934973f, 0.019987956600278217f, -1.2741950499763652f, -0.07851740489327344f, 0.8735224064883065f, 0.9535680852395028f, -1.1310436931234615f, 0.30260427928299005f, -0.6432812658345928f, 0.00018503903591679196f, },
			{ 1.0168765842432597f, 0.9074051766371425f, -1.0417365635354578f, -0.10414319586866373f, 0.6136989989469595f, 0.08708216262529388f, -0.17809664702972364f, -0.582920197527257f, 0.8323114326765764f, -0.17767917734758723f, 0.31840792427811776f, -0.14626683028809404f, },
			{ -1.3312056461473734f, 1.7162130072457586f, 1.2455520963378999f, 0.9333836408015141f, -1.2589367846458575f, 0.8076735030887833f, 0.6609830400573959f, 1.2295588433050466f, 0.6336177240685944f, -0.8461628533180764f, -0.3178135958027358f, -0.6637929909198224f, },
			{ -0.9880205221942007f, -2.2844439005078367f, -0.5162865456227548f, -0.722907066710871f, 1.025165061362178f, 0.26053288309873424f, 0.516904159821145f, -0.2785383970347277f, 0.4247430606936864f, -1.122236951714818f, 1.5521160470183955f, 1.0322123730638566f, },
			{ -0.8204357303491994f, 1.0126189068213711f, 0.5752278250986781f, 0.4932321536772521f, -1.114068315540885f, 0.41151374474938884f, -0.05317312932355234f, -1.184576830758141f, -2.2305846611607403f, -0.5870428741666626f, -0.05733420546405441f, -0.6535716422138564f, },
			{ 0.8004201807191641f, -0.7671637689969776f, 0.3607677199341685f, 0.41292404402105704f, 0.17203647363130375f, 0.42743667149936504f, 0.8935107777688844f, 1.2189366690092163f, 2.4060411601830363f, -0.38806378919749307f, 1.0092185935968154f, -2.1143241572304f, },
			{ 0.4418332181899288f, 0.8963098716860883f, -1.3181434340130682f, 0.6267662226760237f, -0.34164462795383643f, -0.7812241713416036f, -0.5125642509003671f, -1.1284719714686322f, 1.7730156368294994f, 0.47376579927708795f, -1.4159516350004513f, 1.467034447842088f, },
			{ 0.6702499466955174f, -1.0866596363666394f, -1.6962602010376042f, 0.11224086048191191f, 0.023835892873507434f, -2.128456617004704f, -0.7098736361194866f, 0.5363886882414224f, 1.3911045339632981f, -1.4382744050132745f, -0.7713310006688345f, -1.4488454361601977f, },
			{ -0.6367453867953157f, 0.006000246330117187f, -0.9644778034896716f, -1.9709417442057722f, 0.06585680330371564f, 0.26667157314601414f, -1.032917370760953f, -0.5778517954323666f, 1.202616984958666f, -0.856940598027531f, 1.3653670159500662f, -1.3879117051097714f, },
			{ -0.823312086387111f, 1.120019493320961f, 1.55311663181145f, -0.2833119485192546f, -0.09658230422393113f, 1.1223394749139095f, 1.9939328300118204f, 2.0515679420944277f, 0.2273926013724361f, -0.24990492992783678f, 0.8876260959758543f, -0.5195317263425844f, },
			{ 1.2550156746579224f, -1.9033817075568629f, 0.2990540206028145f, 0.6613085807428822f, -1.907861546230383f, -1.8781746672304045f, 0.3003998913071942f, -0.8145898884366449f, -0.7005687205916535f, -0.11114787237046427f, -0.44469314892738016f, -0.5163202715484297f, },
		};
		var ffnUpB = new float[]
		{
			0.5268195660262874f, 1.6094867102794461f, -0.5347960867013055f, -0.9392214622679587f, 0.05328800683077403f, -0.5343944772746517f, -0.13493825515013175f, -1.9155562844632035f, 2.1028005937562604f, 2.102015334043566f, 0.8955057203991021f, 0.24447796368050584f,
		};
		var ffnDownWT = new float[,]
		{
			{ -0.7501256758289891f, -0.2573342625826358f, -0.17195059574884464f, 3.323397906634046f, -0.2120296894153348f, 0.9601511514597083f, 1.7563046171555783f, -0.8358770889152843f, -0.8449642656307342f, 0.07165083787800659f, -0.6417678571422224f, -2.0865674289672516f, },
			{ 0.09406880144562041f, -0.07125553885244983f, -0.6437542682543853f, -0.1723912075183361f, 0.9564549391334319f, -0.14919627018305245f, 1.5952641317739107f, -2.0981818582753853f, 1.003227534175316f, 0.4973606217767939f, 0.6426192325595887f, 0.19423250675558007f, },
			{ -1.4522365716658912f, -1.060507832480397f, -0.11794441963712521f, 1.6311201599134444f, 0.48238536439802304f, 1.3861978155229222f, -0.28028316903700895f, 0.39493216476548376f, -1.0436839047220678f, -0.18431019124372772f, -2.1771125768057176f, -0.6602022751098461f, },
			{ 1.0691234873488105f, 0.5427673418032449f, -0.6476769277009577f, 2.0519874460650214f, -0.16954433119321005f, -0.5793625811542812f, -1.147033025721254f, -0.7957876148733167f, 1.1820007192623958f, 0.8956162076033478f, 0.15473991451688554f, -0.8683792894156708f, },
			{ 1.0935363497517696f, -0.15434478194979898f, -1.3100051139984004f, -0.25887753231461696f, -0.8567271433597247f, 0.9760709110380473f, 1.508901487555591f, -0.700598273450351f, 0.7950012414925455f, -1.3534641816383897f, -0.840230980822416f, -0.9891869006195904f, },
			{ -0.06491833820255845f, -0.0933349522201428f, 1.5200955309513782f, 0.23242018008155776f, -0.9690638101507547f, -0.024784494241557063f, -2.936403234463635f, -2.385020709194674f, 0.010454339143735622f, -0.14339833597550886f, 0.08530999873633503f, 0.4402294031265756f, },
			{ -0.12453509568327982f, 1.6669183661119547f, -0.43833498429692214f, -0.07702544233892183f, 0.33503697274375094f, -0.06177165213263682f, -1.6906843960725f, 0.48728532143708086f, 0.12718405060533938f, 1.7342633638500995f, -0.44498939867474663f, -0.9510101199475669f, },
			{ 0.4197745440198619f, -0.5420409526284742f, -0.8085574618318249f, 1.1585135165034308f, 0.2195946040594563f, -0.5407173956427308f, 1.6034662632559564f, 0.13814734140811005f, -0.3904878015873468f, -1.7150016269610997f, 0.874526114714778f, -0.7544878230670141f, },
			{ 0.8937216388967403f, -0.9067531857611835f, 1.9165325372933435f, -1.6372835861484079f, 0.7751699896143305f, 0.5361656223354984f, -1.7296426036713364f, -1.1496062266049276f, -0.7266283040576533f, -0.7135677973363244f, 1.4975356714989745f, 0.43670849043233917f, },
			{ -1.986898725990354f, -0.4329021697904213f, 0.18536038419298306f, 0.7069667254314276f, 0.0003517332978533371f, -2.1264162320914086f, 1.6102836626917787f, -0.920056946486389f, 0.3344574845921358f, 0.13095802791774278f, -0.3392823224903397f, 1.7996005676038545f, },
			{ -0.04508697897528636f, -0.08861350373183281f, -0.5459735472669436f, 0.3696549859818397f, 0.47327232844771905f, -0.8717568321146669f, -0.17089448265117252f, 0.7101295733993983f, -1.0310887440281757f, -0.1045652311026798f, -0.6158480338699577f, -0.5880035816114038f, },
			{ 1.1819487711400665f, 0.3909913353300458f, 0.5944016368395825f, 2.0073573667661666f, 0.602381422037318f, 1.109943844261619f, 0.21276142185761876f, 1.6415018380955502f, -0.015394821032178992f, 1.3648814916502823f, -0.9405513904795227f, 0.3926181681235894f, },
		};
		var ffnDownB = new float[]
		{
			1.6377410926268572f, 0.22069955194601706f, 0.36671491594652245f, -0.042468052723675966f, 0.2099349610626886f, 1.869201734969229f, 0.8559933330001891f, 1.0909876635220117f, -0.8974051308061993f, -0.16754026990745707f, -0.7161971934274063f, -1.5889818418483315f,
		};
		var ffnLnG = new float[]
		{
			0.37260264774069846f, 0.1909242751220996f, -0.620590190364987f, -0.5110895813258293f, -0.7439820551021338f, -0.7131208595475848f, -2.7185064923950804f, 0.23357881280783982f, 1.2599142210644878f, 1.1536345230283052f, -0.5764906620845378f, 0.5170336526077487f,
		};
		var ffnLnB = new float[]
		{
			0.15271059963001102f, -0.6862567111870019f, -0.771520077335902f, 0.07052908759528514f, -0.5815473973699653f, 0.49666421393201526f, 0.1774084444609198f, 0.48461237143779606f, 1.4820202907532465f, -1.2820572740256437f, -1.62897744925795f, -0.8700826459195213f,
		};
		var headCount = 3;
		var expected = new float[,]
		{
			{ -18.07381762068872f, 29.866739965958608f, -13.72348500711114f, 8.002525083168926f, -23.85621724686734f, -32.26434295264692f, 33.77472802805632f, -29.170106260175583f, -13.496763960003031f, 0.5080231367291841f, -63.61689922705214f, 27.666000554457888f, },
			{ -21.697927745394097f, 29.487303476918164f, -12.47260157688654f, 9.352939885423002f, -26.255314170448838f, -31.052763438643872f, 38.2383785962867f, -28.17032429423469f, -14.361066072775762f, -0.29906179355861084f, -64.87154223859528f, 26.593545210222718f, },
			{ -8.728891183412443f, 17.895150646808553f, -56.540903539920485f, 35.71405290030045f, 20.682711721579274f, -7.95423720779214f, -20.698270050287576f, 19.00295058750639f, -11.289967754639784f, -0.5516241808981324f, 40.09172060872204f, 24.947509077050057f, },
		};

		var layer = new TransformerBlock(
			new(mhaLnG, mhaLnB),
			new(new(mhaUpWT, mhaUpB), new(headCount), new(mhaDownWT, mhaDownB)),
			new(ffnLnG, ffnLnB),
			new(new(ffnUpW, ffnUpB), new(ffnDownWT, ffnDownB)));

		var output = layer.Forward(inputs).ToArray();

		Assert.That(output, Is.EqualTo(expected).Within(0.0001f));
	}
}